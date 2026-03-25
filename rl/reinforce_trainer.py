# rl/trainer.py

import torch
import torch.optim as optim


class RLTrainer:
    def __init__(self, env, policy, sampler, memory_bank, lr=1e-3):
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.memory_bank = memory_bank
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def state_to_tensor(self, state):
        return torch.tensor(state["task_embedding"], dtype=torch.float32).unsqueeze(0)

    async def train_one_episode(self):
        state = self.env.reset()
        state_tensor = self.state_to_tensor(state)

        policy_output = self.policy(state_tensor)
        action, logprob = self.sampler.sample(policy_output)

        _, reward, _, info = await self.env.step(action)

        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        loss = -(logprob * reward_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_memory(info, reward)

        result = info["result"]
        metrics = {
            "reward": reward,
            "correct": result["correct"],
            "total_tokens": result["total_tokens"],
            "used_memory": action["use_memory"],
            "loss": float(loss.item()),
        }
        return metrics

    def update_memory(self, info, reward: float):
        result = info["result"]
        graph_spec = info["graph_spec"]
        task_embedding = info["task_embedding"]

        if reward > 0:
            item = {
                "task_embedding": task_embedding,
                "graph": graph_spec,
                "reward": reward,
                "correct": result["correct"],
                "comm_cost": result["total_tokens"],
            }
            self.memory_bank.add(item)