from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.optim as optim

from utils.v3_trace_utils import build_trace_step, trace_to_local_indices


@dataclass
class PPORouterTransition:
    task_embedding: List[float]
    graph_prior: Dict[str, torch.Tensor]
    candidate_embeddings: torch.Tensor
    visit_counts: List[int]
    last_agent: Optional[int]
    step_idx: int
    selected_local_idx: int
    stop: int
    use_stop_logprob: bool
    use_agent_logprob: bool
    valid_agent_mask: torch.Tensor
    old_logprob: float
    old_value: float


class RouterPPOTrainer:
    def __init__(
        self,
        env,
        router_policy,
        router_sampler,
        memory_bank,
        lr: float = 1e-4,
        max_steps: int = 5,
        add_memory_reward_threshold: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        advantage_norm: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.env = env
        self.router_policy = router_policy
        self.router_sampler = router_sampler
        self.memory_bank = memory_bank
        self.max_steps = max_steps
        self.add_memory_reward_threshold = add_memory_reward_threshold
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.advantage_norm = advantage_norm
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.router_policy.parameters(), lr=lr)

    async def rollout_trace(self, state: Dict) -> Dict:
        task_embedding = torch.tensor(state["task_embedding"], dtype=torch.float32)
        graph_prior = {
            "node_probs": state["graph_prior"]["node_probs"].detach().clone(),
            "edge_probs": state["graph_prior"]["edge_probs"].detach().clone(),
        }
        candidate_embeddings = state["candidate_agent_embeddings"]
        if candidate_embeddings is None:
            raise ValueError("candidate_agent_embeddings must be provided for PPO RouterPolicy scoring.")

        trace: List[Dict] = []
        transitions: List[PPORouterTransition] = []
        visit_counts = [0 for _ in range(len(state["agent_pool"]))]
        last_agent = None

        for step_idx in range(self.max_steps):
            visit_counts_before = list(visit_counts)
            state_repr = self.router_policy.encode_state(
                task_embedding=task_embedding,
                graph_prior=graph_prior,
                visit_counts=visit_counts_before,
                last_agent=last_agent,
                step_idx=step_idx,
                max_steps=self.max_steps,
            )
            policy_output = self.router_policy(state_repr, candidate_embeddings=candidate_embeddings)
            valid_mask = self.router_sampler.build_valid_agent_mask(graph_prior, last_agent, trace)
            action = self.router_sampler.sample(policy_output, valid_mask, trace)

            transitions.append(
                PPORouterTransition(
                    task_embedding=list(state["task_embedding"]),
                    graph_prior={
                        "node_probs": graph_prior["node_probs"].detach().clone(),
                        "edge_probs": graph_prior["edge_probs"].detach().clone(),
                    },
                    candidate_embeddings=candidate_embeddings.detach().clone(),
                    visit_counts=visit_counts_before,
                    last_agent=last_agent,
                    step_idx=step_idx,
                    selected_local_idx=int(action["selected_local_idx"]),
                    stop=int(action["stop"]),
                    use_stop_logprob=bool(action["use_stop_logprob"]),
                    use_agent_logprob=bool(action["use_agent_logprob"]),
                    valid_agent_mask=valid_mask.detach().clone(),
                    old_logprob=float(action["logprob"].detach().item()),
                    old_value=float(policy_output["state_value"].detach().item()),
                )
            )

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

        reward, info = await self.env.execute_trace(trace)
        returns, advantages = self._compute_returns_and_advantages(transitions, float(reward), normalize_advantage=False)
        return {
            "trace": trace,
            "reward": reward,
            "info": info,
            "transitions": transitions,
            "returns": returns,
            "advantages": advantages,
        }

    def _compute_returns_and_advantages(self, transitions: List[PPORouterTransition], final_reward: float,
                                        normalize_advantage: bool = True):
        if len(transitions) == 0:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)

        rewards = [0.0 for _ in transitions]
        rewards[-1] = float(final_reward)
        values = [t.old_value for t in transitions]

        returns = []
        advantages = []
        next_value = 0.0
        next_advantage = 0.0
        for t in reversed(range(len(transitions))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            next_advantage = delta + self.gamma * self.gae_lambda * next_advantage
            advantages.append(next_advantage)
            ret = next_advantage + values[t]
            returns.append(ret)
            next_value = values[t]

        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32)
        if normalize_advantage and self.advantage_norm and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns, advantages

    def _evaluate_transition(self, transition: PPORouterTransition):
        task_embedding = torch.tensor(transition.task_embedding, dtype=torch.float32)
        graph_prior = {
            "node_probs": transition.graph_prior["node_probs"],
            "edge_probs": transition.graph_prior["edge_probs"],
        }
        candidate_embeddings = transition.candidate_embeddings

        state_repr = self.router_policy.encode_state(
            task_embedding=task_embedding,
            graph_prior=graph_prior,
            visit_counts=transition.visit_counts,
            last_agent=transition.last_agent,
            step_idx=transition.step_idx,
            max_steps=self.max_steps,
        )
        policy_output = self.router_policy(state_repr, candidate_embeddings=candidate_embeddings)
        eval_out = self.router_sampler.evaluate_action(
            policy_output=policy_output,
            valid_agent_mask=transition.valid_agent_mask,
            selected_local_idx=transition.selected_local_idx,
            stop=transition.stop,
            use_stop_logprob=transition.use_stop_logprob,
            use_agent_logprob=transition.use_agent_logprob,
        )
        return eval_out["logprob"], eval_out["entropy"], policy_output["state_value"]

    def ppo_update(self, transitions: List[PPORouterTransition], returns: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        if len(transitions) == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }

        old_logprobs = torch.tensor([t.old_logprob for t in transitions], dtype=torch.float32)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0

        for _ in range(self.ppo_epochs):
            new_logprobs = []
            entropies = []
            values = []
            for tr in transitions:
                logprob, entropy, value = self._evaluate_transition(tr)
                new_logprobs.append(logprob.squeeze())
                entropies.append(entropy.squeeze())
                values.append(value.squeeze())

            new_logprobs = torch.stack(new_logprobs)
            entropies = torch.stack(entropies)
            values = torch.stack(values)

            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_bonus = entropies.mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router_policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = (old_logprobs - new_logprobs).mean().abs()
                clip_fraction = ((ratios - 1.0).abs() > self.clip_eps).float().mean()

            total_loss += float(loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_bonus.item())
            total_approx_kl += float(approx_kl.item())
            total_clip_fraction += float(clip_fraction.item())

        denom = float(max(1, self.ppo_epochs))
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_approx_kl / denom,
            "clip_fraction": total_clip_fraction / denom,
        }

    async def train_one_episode(self) -> Dict:
        state = self.env.reset()
        episode = await self.rollout_trace(state)
        ppo_stats = self.ppo_update(episode["transitions"], episode["returns"], episode["advantages"])

        self.update_memory(episode["info"], episode["reward"])
        result = episode["info"]["result"]
        route_stats = episode["info"]["route_stats"]
        return {
            "task_id": state.get("task_id"),
            "reward": float(episode["reward"]),
            "correct": int(result["correct"]),
            "total_tokens": float(result["total_tokens"]),
            "steps": int(route_stats["steps"]),
            "deadloops": int(route_stats["deadloops"]),
            "trace": episode["trace"],
            "trace_local": trace_to_local_indices(episode["trace"]),
            **ppo_stats,
        }

    async def train_batch(self, batch_episodes: int = 16) -> Dict:
        episodes = []

        for _ in range(batch_episodes):
            state = self.env.reset()
            episode = await self.rollout_trace(state)
            episode["state"] = state
            episodes.append(episode)

        batch_transitions, batch_returns, batch_advantages = self._merge_episode_batch(episodes)
        ppo_stats = self.ppo_update(batch_transitions, batch_returns, batch_advantages)

        # memory 还是按 episode 写回
        for ep in episodes:
            self.update_memory(ep["info"], ep["reward"])

        rewards = [float(ep["reward"]) for ep in episodes]
        corrects = [int(ep["info"]["result"]["correct"]) for ep in episodes]
        tokens = [float(ep["info"]["result"]["total_tokens"]) for ep in episodes]
        steps = [int(ep["info"]["route_stats"]["steps"]) for ep in episodes]
        deadloops = [int(ep["info"]["route_stats"]["deadloops"]) for ep in episodes]

        return {
            "batch_episodes": len(episodes),
            "num_transitions": len(batch_transitions),
            "reward_mean": sum(rewards) / max(1, len(rewards)),
            "correct_rate": sum(corrects) / max(1, len(corrects)),
            "token_mean": sum(tokens) / max(1, len(tokens)),
            "steps_mean": sum(steps) / max(1, len(steps)),
            "deadloops_mean": sum(deadloops) / max(1, len(deadloops)),
            **ppo_stats,
        }
    
    def update_memory(self, info: Dict, reward: float) -> None:
        result = info["result"]
        trace = info["trace"]
        if reward < self.add_memory_reward_threshold and int(result["correct"]) == 0:
            return

        item = {
            "task_embedding": info["task_embedding"],
            "task_text": info["task_text"],
            "agent_pool": info["agent_pool"],
            "trace": trace,
            "selected_trace": info["selected_trace"],
            "support_graph": info["support_graph"],
            "support_edges_semantic": info.get("support_edges_semantic", []),
            "execution_graph": info.get("execution_graph", {}),
            "reward": float(reward),
            "correct": int(result["correct"]),
            "token_cost": float(result["total_tokens"]),
            "steps": int(len(info["selected_trace"])),
            "num_agents": len(info["agent_pool"]),
        }
        self.memory_bank.add(item)

    def _merge_episode_batch(self, episodes: List[Dict]):
        all_transitions: List[PPORouterTransition] = []
        returns_list = []
        advantages_list = []

        for ep in episodes:
            if len(ep["transitions"]) == 0:
                continue
            all_transitions.extend(ep["transitions"])
            returns_list.append(ep["returns"])
            advantages_list.append(ep["advantages"])

        if len(all_transitions) == 0:
            return [], torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)

        returns = torch.cat(returns_list, dim=0)
        advantages = torch.cat(advantages_list, dim=0)

        if self.advantage_norm and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return all_transitions, returns, advantages
