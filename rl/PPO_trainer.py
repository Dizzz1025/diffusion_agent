import torch
import torch.nn.functional as F
import torch.optim as optim

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.old_logprobs.clear()
        self.rewards.clear()
        self.values.clear()


class PPOTrainer:
    def __init__(
        self,
        env,
        policy,
        sampler,
        memory_bank,
        lr=1e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=16,
        rollout_size=32,
    ):
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.memory_bank = memory_bank

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_size = rollout_size

        self.buffer = RolloutBuffer()

    def state_to_tensor(self, state):
        return torch.tensor(state["task_embedding"], dtype=torch.float32)

    async def collect_rollouts(self):
        self.buffer.clear()
        metrics = []

        for _ in range(self.rollout_size):
            state = self.env.reset()
            state_tensor = self.state_to_tensor(state).unsqueeze(0)   # [1, D]

            with torch.no_grad():
                policy_output = self.policy(state_tensor)
                action, old_logprob = self.sampler.sample(policy_output)
                value = policy_output["value"]  # [1]

            _, reward, _, info = await self.env.step({
                "use_memory": int(action["use_memory"][0][0].item()),
                "node_mask": action["node_mask"][0].int().tolist(),
                "edge_mask": action["edge_mask"][0].int().tolist(),
            })

            self.buffer.states.append(state_tensor.squeeze(0))  # [D]
            self.buffer.actions.append({
                "use_memory": action["use_memory"].detach(),
                "node_mask": action["node_mask"].detach(),
                "edge_mask": action["edge_mask"].detach(),
            })
            self.buffer.old_logprobs.append(old_logprob.detach())  # [1]
            self.buffer.rewards.append(torch.tensor([reward], dtype=torch.float32))
            self.buffer.values.append(value.detach())              # [1]

            self.update_memory(info, reward)

            metrics.append({
                "reward": reward,
                "correct": info["result"]["correct"],
                "total_tokens": info["result"]["total_tokens"],
            })

        return metrics

    def update_memory(self, info, reward):
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

    def prepare_batch_tensors(self):
        states = torch.stack(self.buffer.states, dim=0)                        # [B, D]
        old_logprobs = torch.cat(self.buffer.old_logprobs, dim=0)              # [B]
        rewards = torch.cat(self.buffer.rewards, dim=0)                        # [B]
        values = torch.cat(self.buffer.values, dim=0)                          # [B]

        returns = rewards
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, old_logprobs, returns, advantages

    def ppo_update(self):
        states, old_logprobs, returns, advantages = self.prepare_batch_tensors()

        num_samples = states.size(0)
        indices = torch.arange(num_samples)

        for _ in range(self.ppo_epochs):
            perm = indices[torch.randperm(num_samples)]

            for start in range(0, num_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]

                batch_states = states[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                policy_output = self.policy(batch_states)

                batch_actions = {
                    "use_memory": torch.cat([self.buffer.actions[i]["use_memory"] for i in idx.tolist()], dim=0),
                    "node_mask": torch.cat([self.buffer.actions[i]["node_mask"] for i in idx.tolist()], dim=0),
                    "edge_mask": torch.cat([self.buffer.actions[i]["edge_mask"] for i in idx.tolist()], dim=0),
                }

                new_logprobs, entropy = self.sampler.evaluate_actions(policy_output, batch_actions)
                values = policy_output["value"]

                ratio = torch.exp(new_logprobs - batch_old_logprobs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    async def train_one_iteration(self):
        metrics = await self.collect_rollouts()
        self.ppo_update()

        avg_reward = sum(m["reward"] for m in metrics) / len(metrics)
        avg_correct = sum(m["correct"] for m in metrics) / len(metrics)
        avg_tokens = sum(m["total_tokens"] for m in metrics) / len(metrics)

        return {
            "avg_reward": avg_reward,
            "avg_correct": avg_correct,
            "avg_tokens": avg_tokens,
            "num_episodes": len(metrics),
        }