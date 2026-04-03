from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

import torch

from env.reward_v3 import V3RewardCalculator
from env.v3_task_env import MultiAgentGraphV3Env
from mas.executor import MultiAgentExecutor
from memory.trajectory_memory_bank import TrajectoryMemoryBank
from policy.router_policy import RouterPolicy
from policy.router_sampler import RouterSampler
from rl.router_ppo_trainer import RouterPPOTrainer
from tasks.math_adapter import MathAdapter
from utils.v3_trace_utils import (
    bootstrap_candidate_traces,
    build_task_agent_runtime,
    build_trace_from_indices,
    trace_to_graph_spec,
)

from GDesigner.llm.profile_embedding import get_sentence_embedding
import os

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
    save_path: str = None
):
    print("[V3-MATH] bootstrap memory bank from candidate traces.")
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
                    # "task_embedding": task_embedding,
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
        top_items = scored[:keep_top_k_per_task]
        memory_bank.add_many(top_items)
        # 再立刻 append 到磁盘
        if save_path is not None:
            with open(save_path, "a", encoding="utf-8") as f:
                for item in top_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())



async def main(args: argparse.Namespace):
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent_names = [x.strip() for x in args.agent_names.split(",") if x.strip()]
    node_roles = [x.strip() for x in args.node_roles.split(",") if x.strip()]
    if len(agent_names) != len(node_roles):
        raise ValueError("agent_names and node_roles must have the same length.")
    node_kwargs = [{"role": role} for role in node_roles]

    adapter = MathAdapter()
    tasks = adapter.load_tasks(args.dataset_json)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    executor = MultiAgentExecutor(
        domain="math",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        num_rounds=args.num_rounds,
        task_adapter=adapter,
        optimized_spatial=False,
        optimized_temporal=False,
        node_kwargs=node_kwargs,
    )

    reward_calculator = V3RewardCalculator(
        alpha_correctness=args.alpha_correctness,
        beta_tokens=args.beta_tokens,
        gamma_steps=args.gamma_steps,
        delta_deadloop=args.delta_deadloop,
        min_steps_free=args.min_steps_free,
        short_trace_penalty=args.short_trace_penalty,
        short_trace_threshold=args.short_trace_threshold,
        use_binary_correctness=True,
    )

    agent_profile_embeddings = build_agent_profile_embeddings(node_kwargs, agent_names)

    memory_bank = TrajectoryMemoryBank(
        max_size=args.memory_max_size,
        corrective_max_size=args.memory_corrective_max_size,
        prototype_match_threshold=args.prototype_match_threshold,
        positive_reward_threshold=args.positive_reward_threshold,
        corrective_gap_threshold=args.corrective_gap_threshold,
        corrective_low_reward_threshold=args.corrective_low_reward_threshold,
    )
    memory_path = str(save_dir / "memory_bootstrap.jsonl")
    await bootstrap_memory_bank(
        tasks=tasks,
        task_adapter=adapter,
        executor=executor,
        reward_calculator=reward_calculator,
        memory_bank=memory_bank,
        default_agent_names=agent_names,
        default_node_kwargs=node_kwargs,
        bootstrap_task_limit=args.bootstrap_task_limit,
        keep_top_k_per_task=args.keep_top_k_per_task,
        save_path=memory_path
    )

    # if args.memory_bootstrap_path:
    #     memory_bank.load_jsonl(args.memory_bootstrap_path)
    # else:
    #     memory_bank.export_jsonl(str(save_dir / "memory_bootstrap.jsonl"))

    task_dim = len(memory_bank.items[0]["task_embedding"]) if len(memory_bank) > 0 else 384

    router_sampler = RouterSampler(node_threshold=args.node_threshold, edge_threshold=args.edge_threshold)
    env = MultiAgentGraphV3Env(
        tasks=tasks,
        task_adapter=adapter,
        executor=executor,
        reward_calculator=reward_calculator,
        memory_bank=memory_bank,
        graph_generator=None,
        agent_profile_embeddings=agent_profile_embeddings,
        top_k_memory=args.top_k_memory,
        node_threshold=router_sampler.node_threshold,
        edge_threshold=router_sampler.edge_threshold,
    )

    router_policy = RouterPolicy(task_dim=task_dim, agent_dim=agent_profile_embeddings.size(-1), hidden_dim=args.hidden_dim)
    trainer = RouterPPOTrainer(
        env=env,
        router_policy=router_policy,
        router_sampler=router_sampler,
        memory_bank=memory_bank,
        lr=args.lr,
        max_steps=args.max_steps,
        add_memory_reward_threshold=args.add_memory_reward_threshold,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
    )

    history = []
    best_reward_mean = float("-inf")

    print("[V3-MATH] start PPO training.")
    for update_idx in range(args.num_updates):
        metrics = await trainer.train_batch(batch_episodes=args.batch_episodes)

        record = {
            "update": update_idx,
            "batch_episodes": int(metrics["batch_episodes"]),
            "num_transitions": int(metrics["num_transitions"]),
            "reward_mean": float(metrics["reward_mean"]),
            "correct_rate": float(metrics["correct_rate"]),
            "token_mean": float(metrics["token_mean"]),
            "steps_mean": float(metrics["steps_mean"]),
            "deadloops_mean": float(metrics["deadloops_mean"]),
            "loss": float(metrics["loss"]),
            "policy_loss": float(metrics["policy_loss"]),
            "value_loss": float(metrics["value_loss"]),
            "entropy": float(metrics["entropy"]),
            "approx_kl": float(metrics["approx_kl"]),
            "clip_fraction": float(metrics["clip_fraction"]),
            "memory_size": len(memory_bank),
        }
        history.append(record)

        if update_idx % args.log_every == 0:
            print(
                f"[router-math] update={update_idx} | "
                f"reward_mean={record['reward_mean']:.4f} | "
                f"correct_rate={record['correct_rate']:.4f} | "
                f"token_mean={record['token_mean']:.1f} | "
                f"steps_mean={record['steps_mean']:.2f} | "
                f"deadloops_mean={record['deadloops_mean']:.2f} | "
                f"loss={record['loss']:.4f} | "
                f"kl={record['approx_kl']:.4f}"
            )

        if record["reward_mean"] > best_reward_mean:
            best_reward_mean = record["reward_mean"]
            torch.save(
                {
                    "update": update_idx,
                    "router_policy_state_dict": router_policy.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "best_reward_mean": best_reward_mean,
                    "history": history,
                    "config": vars(args),
                },
                ckpt_dir / "router_best.pt",
            )

        with (save_dir / "train_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if (update_idx + 1) % args.export_memory_every == 0:
            memory_bank.export_jsonl(str(save_dir / f"memory_update_{update_idx + 1}.jsonl"))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, default="my_datasets/MATH/wrong.jsonl")
    parser.add_argument("--llm_name", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--save_dir", type=str, default="results/v4_math")
    # parser.add_argument("--memory_bootstrap_path", type=str, default="/root/autodl-tmp/diffusion_agent/results/v4_math/memory_bootstrap.jsonl")
    parser.add_argument("--memory_bootstrap_path", type=str, default="/root/autodl-tmp/diffusion_agent/results/v4_math/memory_bootstrap.jsonl")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--max_tasks", type=int, default=-1)

    parser.add_argument("--agent_names", type=str, default="MathSolver,MathSolver,MathSolver,MathSolver")
    parser.add_argument("--node_roles", type=str, default="MathSolver,ProblemDecomposer,CalculationChecker,ProgrammingExpert")

    parser.add_argument("--memory_max_size", type=int, default=300)
    parser.add_argument("--bootstrap_task_limit", type=int, default=100)
    parser.add_argument("--keep_top_k_per_task", type=int, default=2)
    parser.add_argument("--top_k_memory", type=int, default=5)

    parser.add_argument("--memory_corrective_max_size", type=int, default=200)
    parser.add_argument("--prototype_match_threshold", type=float, default=0.72)
    parser.add_argument("--positive_reward_threshold", type=float, default=0.0)
    parser.add_argument("--corrective_gap_threshold", type=float, default=0.35)
    parser.add_argument("--corrective_low_reward_threshold", type=float, default=0.0)

    parser.add_argument("--node_threshold", type=float, default=0.35)
    parser.add_argument("--edge_threshold", type=float, default=0.35)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--add_memory_reward_threshold", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_updates", type=int, default=100)
    parser.add_argument("--batch_episodes", type=int, default=16)

    parser.add_argument("--alpha_correctness", type=float, default=2.0)
    parser.add_argument("--beta_tokens", type=float, default=0.00005)
    parser.add_argument("--gamma_steps", type=float, default=0.02)
    parser.add_argument("--delta_deadloop", type=float, default=0.20)
    parser.add_argument("--min_steps_free", type=int, default=2)
    parser.add_argument("--short_trace_penalty", type=float, default=0.15)
    parser.add_argument("--short_trace_threshold", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--export_memory_every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
