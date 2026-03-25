# scripts/train.py

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
from rl.PPO_trainer import PPOTrainer  # 假设你已经实现了 PPOTrainer

async def main():
    # ------------------------------
    # 1. 配置数据和模型
    # ------------------------------
    dataset_json = "my_datasets/gsm8k/gsm8k_train.jsonl"
    llm_name = "/home/zhangdi24/Qwen2.5-7B-Instruct"
    domain = "gsm8k"
    decision_method = "FinalRefer"
    num_rounds = 1

    # ------------------------------
    # 2. 初始化 Agent 配置
    # ------------------------------
    agent_names = ["MathSolver", "MathSolver", "MathSolver", "MathSolver"]
    node_kwargs = [
        {'role': 'MathSolver'},
        {'role': 'ProblemDecomposer'},
        {'role': 'CalculationChecker'},
        {'role': 'ProgrammingExpert'},
    ]

    # ------------------------------
    # 3. 数据加载和环境构建
    # ------------------------------
    adapter = GSM8KAdapter()
    tasks = adapter.load_tasks(dataset_json)

    memory_bank = MemoryBank(max_size=50)

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

    # ------------------------------
    # 4. 初始化图策略和采样器
    # ------------------------------
    policy = GraphPolicy(
        state_dim=384,
        hidden_dim=256,
        num_agents=len(agent_names),
    )

    sampler = GraphSampler()

    # ------------------------------
    # 5. 初始化 PPO Trainer
    # ------------------------------
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        sampler=sampler,
        memory_bank=memory_bank,
        lr=1e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=16,
        rollout_size=32,  # 每次收集32道题作为一个batch
    )

    # ------------------------------
    # 6. 训练保存配置
    # ------------------------------
    history = []
    save_dir = "results/train_vis"
    os.makedirs(save_dir, exist_ok=True)

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_reward = float('-inf')

    # ------------------------------
    # 7. 主训练循环
    # ------------------------------
    num_iterations = 100
    for it in range(num_iterations):
        metrics = await trainer.train_one_iteration()  # PPO：收集一批 rollout + update

        # 保存训练历史
        record = {
            "iteration": it,
            "avg_reward": float(metrics["avg_reward"]),
            "avg_correct": int(metrics["avg_correct"]),
            "avg_tokens": float(metrics["avg_tokens"]),
        }
        history.append(record)

        with open(os.path.join(save_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # 打印日志
        if it % 5 == 0:
            print(
                f"Iter={it} | "
                f"avg_reward={metrics['avg_reward']:.4f} | "
                f"avg_correct={metrics['avg_correct']} | "
                f"avg_tokens={metrics['avg_tokens']:.2f}"
            )

        # 保存最佳模型
        if metrics["avg_reward"] > best_reward:
            best_reward = metrics["avg_reward"]
            best_ckpt = {
                "iteration": it,
                "policy_state_dict": policy.state_dict(),
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
                }
            }
            torch.save(best_ckpt, os.path.join(ckpt_dir, "best.pt"))

if __name__ == "__main__":
    asyncio.run(main())