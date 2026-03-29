from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List

import torch

from env.reward_v3 import V3RewardCalculator
from env.v3_task_env import MultiAgentGraphV3Env
from mas.executor import MultiAgentExecutor
from memory.trajectory_memory_bank import TrajectoryMemoryBank
from policy.graph_generator import GraphGenerator, graph_generator_loss
from policy.router_policy import RouterPolicy
from policy.router_sampler import RouterSampler
from rl.router_ppo_trainer import RouterPPOTrainer
from tasks.gsm8k_adapter import GSM8KAdapter
from utils.v3_trace_utils import (
    bootstrap_candidate_traces,
    build_task_agent_runtime,
    build_trace_from_indices,
    trace_to_graph_spec,
)

from GDesigner.llm.profile_embedding import get_sentence_embedding
import debugpy

debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach on port 5678...")
debugpy.wait_for_client()
print("Debugger attached.")

def build_agent_profile_embeddings(node_kwargs: List[Dict], fallback_names: List[str]) -> torch.Tensor:
    texts = []
    for idx, name in enumerate(fallback_names):
        role_text = name
        if idx < len(node_kwargs) and isinstance(node_kwargs[idx], dict):
            role_text = node_kwargs[idx].get("role", name)
        texts.append(role_text)
    embs = [get_sentence_embedding(text) for text in texts]
    return torch.tensor(embs, dtype=torch.float32).unsqueeze(0)


async def bootstrap_memory_bank(
    tasks,
    task_adapter,
    executor,
    reward_calculator,
    memory_bank,
    default_agent_names,
    default_node_kwargs,
    bootstrap_task_limit: int = 30,
    keep_top_k_per_task: int = 2,
):
    print("[V3] Step 1-3: bootstrap candidate traces, execute them directly, and write strong trajectories into memory.")
    num_agents = len(default_agent_names)
    candidates = bootstrap_candidate_traces(num_agents)

    for task in tasks[:bootstrap_task_limit]:
        task_embedding = get_sentence_embedding(task_adapter.get_task_text(task)).tolist()
        agent_names, node_kwargs, agent_pool, _ = build_task_agent_runtime(task, default_agent_names, default_node_kwargs)
        scored = []
        for trace_indices in candidates:
            trace = build_trace_from_indices(trace_indices, agent_pool)
            result = await executor.run_trace(
                task,
                trace,
                graph_prior=None,
                agent_names_override=agent_names,
                node_kwargs_override=node_kwargs,
            )
            reward = reward_calculator.compute(result, {"steps": len(trace_indices), "deadloops": 0})
            scored.append(
                {
                    "task_embedding": task_embedding,
                    "task_text": task["task_text"],
                    "agent_pool": agent_pool,
                    "trace": trace,
                    "selected_trace": trace_indices,
                    "support_graph": trace_to_graph_spec(trace, num_agents),
                    "execution_graph": result.get("graph", {}),
                    "reward": reward,
                    "correct": int(result["correct"]),
                    "token_cost": float(result["total_tokens"]),
                    "steps": len(trace_indices),
                    "num_agents": num_agents,
                }
            )

        scored.sort(key=lambda x: (x["correct"], x["reward"]), reverse=True)
        memory_bank.add_many(scored[:keep_top_k_per_task])


async def train_graph_generator_from_memory(
    graph_generator: GraphGenerator,
    memory_bank: TrajectoryMemoryBank,
    default_agent_profile_embeddings: torch.Tensor,
    current_agent_pool,
    epochs: int = 20,
    top_k: int = 5,
    lr: float = 1e-3,
):
    if len(memory_bank) == 0:
        print("[V3] memory bank is empty; skip graph generator training.")
        return

    print(f"[V3] Step 5: training graph generator from {len(memory_bank)} memory items.")
    optimizer = torch.optim.Adam(graph_generator.parameters(), lr=lr)
    graph_generator.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for idx, item in enumerate(memory_bank.items):
            item_agent_pool = item.get("agent_pool", current_agent_pool) ## TODO: 感觉写反了
            item_agent_names = [x.get("agent_name", f"Agent{i}") for i, x in enumerate(item_agent_pool)]
            item_node_kwargs = [
                {"role": x.get("agent_role", x.get("agent_name", f"Agent{i}")), "desc": x.get("agent_desc", x.get("agent_role", x.get("agent_name", f"Agent{i}")))}
                for i, x in enumerate(item_agent_pool)
            ]
            _, _, item_agent_pool, item_agent_profile_embeddings = build_task_agent_runtime(
                {"agent_pool": item_agent_pool},
                item_agent_names,
                item_node_kwargs,
            )
            labels = memory_bank.build_soft_labels(
                task_embedding=item["task_embedding"],
                current_agent_pool=item_agent_pool,
                top_k=top_k,
                fallback_item=item,
                exclude_index=idx, # 构建标签时排除当前样本，避免自己给自己做标签
            )
            summary = memory_bank.summarize(
                task_embedding=item["task_embedding"],
                current_agent_pool=item_agent_pool,
                top_k=top_k,
            )
            summary_tensors = {
                "node_prior": torch.tensor(summary["node_prior"], dtype=torch.float32).unsqueeze(0),
                "edge_prior": torch.tensor(summary["edge_prior"], dtype=torch.float32).unsqueeze(0),
            }
            task_tensor = torch.tensor(item["task_embedding"], dtype=torch.float32).unsqueeze(0)
            node_targets = torch.tensor(labels["node_targets"], dtype=torch.float32).unsqueeze(0)
            edge_targets = torch.tensor(labels["edge_targets"], dtype=torch.float32).unsqueeze(0)

            outputs = graph_generator(task_tensor, item_agent_profile_embeddings, memory_summary=summary_tensors)
            loss = graph_generator_loss(outputs, node_targets, edge_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        print(f"  [graph-generator] epoch={epoch + 1}/{epochs} loss={total_loss / max(1, num_batches):.4f}")


async def main():
    dataset_json = "my_datasets/gsm8k/gsm8k_train.jsonl"
    # llm_name = "/home/zhangdi24/Qwen2.5-7B-Instruct"
    llm_name = 'Meta-Llama-3.1-8B-Instruct'
    domain = "gsm8k"
    decision_method = "FinalRefer"
    num_rounds = 1

    agent_names = ["MathSolver", "MathSolver", "MathSolver", "MathSolver"]
    node_kwargs = [
        {"role": "MathSolver"},
        {"role": "ProblemDecomposer"},
        {"role": "CalculationChecker"},
        {"role": "ProgrammingExpert"},
    ]

    save_dir = Path("results/v3")
    ckpt_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    adapter = GSM8KAdapter()
    tasks = adapter.load_tasks(dataset_json)

    executor = MultiAgentExecutor(
        domain=domain,
        llm_name=llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        num_rounds=num_rounds,
        task_adapter=adapter,
        optimized_spatial=False,
        optimized_temporal=False,
        node_kwargs=node_kwargs,
    )

    reward_calculator = V3RewardCalculator(
        alpha_correctness=1.0,
        beta_tokens=0.001,
        gamma_steps=0.05,
        delta_deadloop=0.10,
    )

    agent_profile_embeddings = build_agent_profile_embeddings(node_kwargs, agent_names)

    memory_bank = TrajectoryMemoryBank(max_size=10)
    # await bootstrap_memory_bank(
    #     tasks=tasks,
    #     task_adapter=adapter,
    #     executor=executor,
    #     reward_calculator=reward_calculator,
    #     memory_bank=memory_bank,
    #     default_agent_names=agent_names,
    #     default_node_kwargs=node_kwargs,
    #     bootstrap_task_limit=1,
    #     keep_top_k_per_task=2,
    # )
    # memory_bank.export_jsonl(str(save_dir / "memory_bootstrap.jsonl"))
    memory_bank.load_jsonl(str(save_dir / "memory_bootstrap.jsonl"))

    task_dim = len(memory_bank.items[0]["task_embedding"]) if len(memory_bank) > 0 else 384
    graph_generator = GraphGenerator(
        task_dim=task_dim,
        agent_dim=agent_profile_embeddings.size(-1),
        hidden_dim=256,
        num_agents=len(agent_names),
    )

    await train_graph_generator_from_memory(
        graph_generator=graph_generator,
        memory_bank=memory_bank,
        default_agent_profile_embeddings=agent_profile_embeddings,
        current_agent_pool=build_task_agent_runtime(tasks[0] if tasks else {}, agent_names, node_kwargs)[2],
        epochs=15,
        top_k=5,
        lr=1e-3,
    )
    torch.save(graph_generator.state_dict(), ckpt_dir / "graph_generator.pt")

    router_sampler = RouterSampler(node_threshold=0.35, edge_threshold=0.35)
    env = MultiAgentGraphV3Env(
        tasks=tasks,
        task_adapter=adapter,
        executor=executor,
        reward_calculator=reward_calculator,
        memory_bank=memory_bank,
        graph_generator=graph_generator,
        agent_profile_embeddings=agent_profile_embeddings,
        top_k_memory=5,
        node_threshold=router_sampler.node_threshold,
        edge_threshold=router_sampler.edge_threshold,
    )

    router_policy = RouterPolicy(task_dim=task_dim, agent_dim=agent_profile_embeddings.size(-1), hidden_dim=256)
    trainer = RouterPPOTrainer(
        env=env,
        router_policy=router_policy,
        router_sampler=router_sampler,
        memory_bank=memory_bank,
        lr=1e-4,
        max_steps=5,
        add_memory_reward_threshold=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
    )

    history = []
    best_reward = float("-inf")
    num_episodes = 100

    print("[V3] Step 6-8: train router under graph constraints and execute the selected trace directly.")
    for episode in range(num_episodes):
        metrics = await trainer.train_one_episode()
        record = {
            "episode": episode,
            "reward": float(metrics["reward"]),
            "correct": int(metrics["correct"]),
            "total_tokens": float(metrics["total_tokens"]),
            "steps": int(metrics["steps"]),
            "deadloops": int(metrics["deadloops"]),
            "trace": metrics["trace"],
            "trace_local": metrics["trace_local"],
            "loss": float(metrics["loss"]),
            "policy_loss": float(metrics["policy_loss"]),
            "value_loss": float(metrics["value_loss"]),
            "entropy": float(metrics["entropy"]),
            "approx_kl": float(metrics["approx_kl"]),
            "clip_fraction": float(metrics["clip_fraction"]),
            "memory_size": len(memory_bank),
        }
        history.append(record)

        if episode % 5 == 0:
            print(
                f"[router] ep={episode} | reward={record['reward']:.4f} | "
                f"correct={record['correct']} | tokens={record['total_tokens']:.1f} | "
                f"steps={record['steps']} | deadloops={record['deadloops']} | loss={record['loss']:.4f} | kl={record['approx_kl']:.4f} | trace_local={record['trace_local']}"
            )

        if record["reward"] > best_reward:
            best_reward = record["reward"]
            torch.save(
                {
                    "episode": episode,
                    "router_policy_state_dict": router_policy.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "best_reward": best_reward,
                    "history": history,
                    "config": {
                        "dataset_json": dataset_json,
                        "llm_name": llm_name,
                        "domain": domain,
                        "decision_method": decision_method,
                        "num_rounds": num_rounds,
                        "agent_names": agent_names,
                    },
                },
                ckpt_dir / "router_best.pt",
            )

        with (save_dir / "train_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if (episode + 1) % 20 == 0:
            memory_bank.export_jsonl(str(save_dir / f"memory_episode_{episode + 1}.jsonl"))


if __name__ == "__main__":
    asyncio.run(main())
