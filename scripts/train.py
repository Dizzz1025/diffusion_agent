# scripts/train.py

import asyncio

from tasks.gsm8k_adapter import GSM8KAdapter
from mas.executor import MultiAgentExecutor
from env.reward import RewardCalculator
from env.task_env import MultiAgentGraphEnv
from memory.memory_bank import MemoryBank
from policy.graph_policy import GraphPolicy
from policy.graph_sampler import GraphSampler
from rl.trainer import RLTrainer

import debugpy

debugpy.listen(("0.0.0.0", 5678))   # 监听调试端口 5678
print("Waiting for debugger attach on port 5678...")
debugpy.wait_for_client()           # 等待调试器连上再继续
print("Debugger attached.")

async def main():
    dataset_json = "my_datasets/gsm8k/gsm8k_train.jsonl"
    llm_name = "/home/zhangdi24/Qwen2.5-7B-Instruct"
    domain = "gsm8k"
    decision_method = "FinalRefer"
    num_rounds = 1

    # 先固定 agent 配置
    agent_names = ["MathSolver", "MathSolver", "MathSolver", "MathSolver"]

    adapter = GSM8KAdapter()
    tasks = adapter.load_tasks(dataset_json)

    memory_bank = MemoryBank(max_size=200)

    executor = MultiAgentExecutor(
        domain=domain,
        llm_name=llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        num_rounds=num_rounds,
        task_adapter=adapter,
        optimized_spatial=False,
        optimized_temporal=False,
        node_kwargs=None,
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

    policy = GraphPolicy(
        state_dim=128,
        hidden_dim=256,
        num_agents=len(agent_names),
    )

    sampler = GraphSampler()
    trainer = RLTrainer(env, policy, sampler, memory_bank, lr=1e-3)

    for epoch in range(100):
        metrics = await trainer.train_one_episode()
        if epoch % 10 == 0:
            print(
                f"Epoch={epoch} | "
                f"reward={metrics['reward']:.4f} | "
                f"correct={metrics['correct']} | "
                f"tokens={metrics['total_tokens']} | "
                f"used_memory={metrics['used_memory']} | "
                f"loss={metrics['loss']:.4f}"
            )


if __name__ == "__main__":
    asyncio.run(main())