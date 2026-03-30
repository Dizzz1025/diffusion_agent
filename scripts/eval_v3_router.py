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
from policy.router_policy import RouterPolicy
from policy.router_sampler import RouterSampler
from tasks.gsm8k_adapter import GSM8KAdapter
from utils.v3_trace_utils import build_trace_step, trace_to_local_indices
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


def greedy_action(
    router_sampler: RouterSampler,
    policy_output: Dict[str, torch.Tensor],
    valid_agent_mask: torch.Tensor,
    trace,
) -> Dict:
    """
    测试阶段用确定性选择：
    1) next_agent = masked logits 的 argmax
    2) stop = sigmoid(stop_logit) >= 0.5
    """
    trace_indices = trace_to_local_indices(trace)

    if valid_agent_mask.dim() == 1:
        valid_agent_mask = valid_agent_mask.unsqueeze(0)

    masked_logits = policy_output["next_agent_logits"].clone()
    masked_logits = masked_logits.masked_fill(valid_agent_mask <= 0, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)

    stop_prob = torch.sigmoid(policy_output["stop_logit"].reshape(-1))[0]
    stop = int((stop_prob >= 0.5).item())

    # 第一跳不能直接 stop
    use_stop_logprob = not (router_sampler.force_non_empty_trace and len(trace_indices) == 0)
    if not use_stop_logprob:
        stop = 0

    picked = int(masked_logits[0].argmax().item())
    use_agent_logprob = (stop == 0) or (len(trace_indices) == 0)

    return {
        "stop": stop,
        "next_agent": picked,
        "selected_local_idx": picked,
        "selection_score": float(masked_logits[0, picked].item()),
        "selection_prob": float(probs[0, picked].item()),
        "allowed_by_graph": bool(valid_agent_mask[0, picked].item() > 0),
        "use_stop_logprob": use_stop_logprob,
        "use_agent_logprob": use_agent_logprob,
    }


async def rollout_trace_greedy(
    env: MultiAgentGraphV3Env,
    router_policy: RouterPolicy,
    router_sampler: RouterSampler,
    state: Dict,
    max_steps: int = 5,
) -> Dict:
    """
    基本复用 RouterPPOTrainer.rollout_trace 的逻辑，
    但只做推理，不做 PPO 更新。
    """
    task_embedding = torch.tensor(state["task_embedding"], dtype=torch.float32)
    graph_prior = {
        "node_probs": state["graph_prior"]["node_probs"].detach().clone(),
        "edge_probs": state["graph_prior"]["edge_probs"].detach().clone(),
    }
    candidate_embeddings = state["candidate_agent_embeddings"]
    if candidate_embeddings is None:
        raise ValueError("candidate_agent_embeddings must be provided for router evaluation.")

    trace: List[Dict] = []
    visit_counts = [0 for _ in range(len(state["agent_pool"]))]
    last_agent = None

    with torch.no_grad():
        for step_idx in range(max_steps):
            visit_counts_before = list(visit_counts)

            state_repr = router_policy.encode_state(
                task_embedding=task_embedding,
                graph_prior=graph_prior,
                visit_counts=visit_counts_before,
                last_agent=last_agent,
                step_idx=step_idx,
                max_steps=max_steps,
            )
            policy_output = router_policy(state_repr, candidate_embeddings=candidate_embeddings)

            valid_mask = router_sampler.build_valid_agent_mask(graph_prior, last_agent, trace)
            action = greedy_action(router_sampler, policy_output, valid_mask, trace)

            if action["stop"] == 1 and len(trace) > 0:
                break

            next_agent = int(action["selected_local_idx"])
            trace.append(
                build_trace_step(
                    local_idx=next_agent,
                    agent_pool=state["agent_pool"],
                    step_idx=step_idx,
                    selection_score=action.get("selection_score"),
                    selection_prob=action.get("selection_prob"),
                    allowed_by_graph=action.get("allowed_by_graph"),
                )
            )
            visit_counts[next_agent] += 1
            last_agent = next_agent

    if len(trace) == 0:
        trace = [build_trace_step(0, state["agent_pool"], step_idx=0)]

    reward, info = await env.execute_trace(trace)
    return {
        "trace": trace,
        "trace_local": trace_to_local_indices(trace),
        "reward": float(reward),
        "info": info,
    }


async def evaluate():
    # ===== 这里按你的实际路径改 =====
    dataset_json = "my_datasets/gsm8k/gsm8k_test.jsonl"
    # llm_name = "Meta-Llama-3.1-8B-Instruct"
    llm_name = "/home/zhangdi24/Qwen2.5-7B-Instruct"
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
    ckpt_path = save_dir / "checkpoints" / "router_best.pt"
    memory_path = save_dir / "memory_bootstrap.jsonl"
    output_path = save_dir / "eval_router_greedy.json"

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

    # 直接加载已有 memory
    memory_bank = TrajectoryMemoryBank(max_size=300)
    memory_bank.load_jsonl(str(memory_path))
    if len(memory_bank) == 0:
        raise ValueError(f"memory bank is empty: {memory_path}")

    task_dim = len(memory_bank.items[0]["task_embedding"])

    router_sampler = RouterSampler(node_threshold=0.35, edge_threshold=0.35)
    env = MultiAgentGraphV3Env(
        tasks=tasks,
        task_adapter=adapter,
        executor=executor,
        reward_calculator=reward_calculator,
        memory_bank=memory_bank,
        graph_generator=None,
        agent_profile_embeddings=agent_profile_embeddings,
        top_k_memory=5,
        node_threshold=router_sampler.node_threshold,
        edge_threshold=router_sampler.edge_threshold,
    )

    router_policy = RouterPolicy(
        task_dim=task_dim,
        agent_dim=agent_profile_embeddings.size(-1),
        hidden_dim=256,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    router_policy.load_state_dict(ckpt["router_policy_state_dict"])
    router_policy.eval()

    records = []
    total_correct = 0
    total_tokens = 0.0
    total_steps = 0
    total_reward = 0.0

    print(f"[eval] loaded {len(tasks)} tasks from {dataset_json}")
    print(f"[eval] loaded memory from {memory_path}")
    print(f"[eval] loaded router ckpt from {ckpt_path}")

    for i, task in enumerate(tasks):
        # 因为 env.reset() 内部会 random.choice(self.tasks)
        # 所以这里每次只塞当前 task，保证评估顺序正确
        env.tasks = [task]
        state = env.reset()

        episode = await rollout_trace_greedy(
            env=env,
            router_policy=router_policy,
            router_sampler=router_sampler,
            state=state,
            max_steps=5,
        )

        result = episode["info"]["result"]
        route_stats = episode["info"]["route_stats"]

        correct = int(result["correct"])
        total_correct += correct
        total_tokens += float(result["total_tokens"])
        total_steps += int(route_stats["steps"])
        total_reward += float(episode["reward"])

        record = {
            "idx": i,
            "task_text": task.get("task_text", ""),
            "correct": correct,
            "reward": float(episode["reward"]),
            "total_tokens": float(result["total_tokens"]),
            "steps": int(route_stats["steps"]),
            "deadloops": int(route_stats["deadloops"]),
            "trace_local": episode["trace_local"],
            "trace": episode["trace"],
        }
        records.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
            acc = total_correct / max(1, len(records))
            print(
                f"[eval] {i+1}/{len(tasks)} | "
                f"acc={acc:.4f} | "
                f"avg_tokens={total_tokens / max(1, len(records)):.2f} | "
                f"avg_steps={total_steps / max(1, len(records)):.2f}"
            )

    summary = {
        "num_samples": len(records),
        "accuracy": total_correct / max(1, len(records)),
        "avg_reward": total_reward / max(1, len(records)),
        "avg_tokens": total_tokens / max(1, len(records)),
        "avg_steps": total_steps / max(1, len(records)),
        "ckpt_path": str(ckpt_path),
        "memory_path": str(memory_path),
        "dataset_json": dataset_json,
        "decode_mode": "greedy",
    }

    output = {
        "summary": summary,
        "records": records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n===== EVAL SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(evaluate())