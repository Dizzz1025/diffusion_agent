import asyncio
import os
import json
import torch

from tasks.gsm8k_adapter import GSM8KAdapter
from mas.executor import MultiAgentExecutor
from env.reward import RewardCalculator
from env.task_env import MultiAgentGraphEnv
from memory.memory_bank import MemoryBank
from policy.graph_policy import GraphPolicy
from policy.graph_sampler import GraphSampler

import debugpy

# debugpy.listen(("0.0.0.0", 5678))   # 监听调试端口 5678
# print("Waiting for debugger attach on port 5678...")
# debugpy.wait_for_client()           # 等待调试器连上再继续
# print("Debugger attached.")

async def evaluate_one_task(env, policy, sampler, task):
    """
    对单个 task 做一次测试，不更新参数
    """
    env.current_task = task
    task_text = env.task_adapter.get_task_text(task)
    env.current_task_embedding = env.encode_task(task_text)

    retrieved = []
    if env.memory_bank is not None:
        retrieved = env.memory_bank.retrieve(env.current_task_embedding, top_k=1)
    env.current_retrieved = retrieved

    state = {
        "task": env.current_task,
        "task_embedding": env.current_task_embedding,
        "retrieved": retrieved,
    }

    state_tensor = torch.tensor(state["task_embedding"], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        policy_output = policy(state_tensor)
        action, _ = sampler.sample(policy_output)

    _, reward, _, info = await env.step(action)
    result = info["result"]

    record = {
        "task_id": result["task_id"],
        "task_text": result["task_text"],
        "ground_truth": result["ground_truth"],
        "predict_answer": result["predict_answer"],
        "correct": int(result["correct"]),
        "task_score": float(result["task_score"]),
        "reward": float(reward),
        "total_tokens": float(result["total_tokens"]),
        "prompt_tokens": float(result["prompt_tokens"]),
        "completion_tokens": float(result["completion_tokens"]),
        "latency": float(result["latency"]),
        "used_memory": int(action["use_memory"]),
        "graph_spec": info["graph_spec"],
        "raw_answer": result["raw_answer"],
    }
    return record


async def main():
    # ===== 基本配置 =====
    dataset_json = "my_datasets/gsm8k/gsm8k_test.jsonl"
    ckpt_path = "results/train_vis/checkpoints/best.pt"
    save_dir = "results/eval_vis"
    os.makedirs(save_dir, exist_ok=True)

    # ===== 读取 checkpoint =====
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})

    llm_name = config.get("llm_name", "Meta-Llama-3.1-8B-Instruct")
    domain = config.get("domain", "gsm8k")
    decision_method = config.get("decision_method", "FinalRefer")
    num_rounds = config.get("num_rounds", 1)
    agent_names = config.get(
        "agent_names",
        ["MathSolver", "MathSolver", "MathSolver", "MathSolver"]
    )

    # 这里保持和训练时一致
    node_kwargs = [
        {"role": "MathSolver"},
        {"role": "ProblemDecomposer"},
        {"role": "CalculationChecker"},
        {"role": "ProgrammingExpert"},
    ]

    # ===== 构建任务适配器与测试集 =====
    adapter = GSM8KAdapter()
    tasks = adapter.load_tasks(dataset_json)
    # tasks = tasks[:10]
    # 当前测试脚本先用空 memory bank
    memory_bank = MemoryBank(max_size=50)

    # ===== 构建执行器 =====
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

    # ===== 构建 reward / env =====
    reward_calculator = RewardCalculator(
        alpha_task=1.0,
        gamma_tokens=0.001,
        delta_latency=0.0,
    )

    env = MultiAgentGraphEnv(
        tasks=tasks,
        task_adapter=adapter,
        executor=executor,
        reward_calculator=reward_calculator,
        memory_bank=memory_bank,
    )

    # ===== 构建 policy / sampler =====
    # 这里必须和训练时一致
    policy = GraphPolicy(
        state_dim=384,
        hidden_dim=256,
        num_agents=len(agent_names),
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    sampler = GraphSampler()

    # ===== 开始测试 =====
    results = []
    total_correct = 0
    total_reward = 0.0
    total_tokens = 0.0

    for i, task in enumerate(tasks):
        record = await evaluate_one_task(env, policy, sampler, task)
        results.append(record)

        total_correct += record["correct"]
        total_reward += record["reward"]
        total_tokens += record["total_tokens"]

        if i % 10 == 0:
            print(
                f"[{i}/{len(tasks)}] "
                f"correct={record['correct']} | "
                f"reward={record['reward']:.4f} | "
                f"tokens={record['total_tokens']}"
            )

    # ===== 汇总结果 =====
    num_tasks = len(tasks)
    summary = {
        "num_tasks": num_tasks,
        "accuracy": total_correct / num_tasks if num_tasks > 0 else 0.0,
        "avg_reward": total_reward / num_tasks if num_tasks > 0 else 0.0,
        "avg_tokens": total_tokens / num_tasks if num_tasks > 0 else 0.0,
        "checkpoint": ckpt_path,
        "dataset_json": dataset_json,
    }

    print("\n===== Evaluation Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open(os.path.join(save_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())