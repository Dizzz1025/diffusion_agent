from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class RouterPolicy(nn.Module):
    """Candidate-scoring router policy with a shared actor-critic backbone.

    The policy scores the *current candidate agent set* rather than predicting a
    fixed semantic node id. A value head is added so the same module can be used
    by PPO without introducing a separate critic network.
    """

    def __init__(self, task_dim: int, agent_dim: int, hidden_dim: int = 256):
        super().__init__()
        global_input_dim = task_dim + 7
        candidate_extra_dim = 6
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim + agent_dim + candidate_extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # self.stop_head = nn.Linear(hidden_dim, 1)
        # [MOD] 原来的 stop_head 删除，改成一个可学习的 decision embedding
        self.decision_embedding = nn.Parameter(torch.randn(1, 1, agent_dim) * 0.02)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode_state(
        self,
        task_embedding: torch.Tensor,
        graph_prior: Dict[str, torch.Tensor],
        visit_counts: List[int],
        last_agent: Optional[int],
        step_idx: int,
        max_steps: int,
    ) -> Dict[str, torch.Tensor]:
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)

        batch_size = task_embedding.size(0)
        node_probs = graph_prior["node_probs"]
        edge_probs = graph_prior["edge_probs"]
        if node_probs.dim() == 1:
            node_probs = node_probs.unsqueeze(0)
        if edge_probs.dim() == 2:
            edge_probs = edge_probs.unsqueeze(0)

        num_agents = node_probs.size(-1) # node_probs.size() = (1, 4)
        device = task_embedding.device
        visits = torch.tensor(visit_counts, dtype=torch.float32, device=device).unsqueeze(0)
        if visits.size(-1) != num_agents:
            raise ValueError(f"visit_counts size {visits.size(-1)} does not match num_agents {num_agents}")

        visit_norm = visits / max(1.0, float(max_steps))
        visited_mask = (visits > 0).float()
        last_one_hot = torch.zeros(batch_size, num_agents, device=device)
        last_exists = torch.zeros(batch_size, 1, device=device)
        if last_agent is not None and 0 <= int(last_agent) < num_agents:
            last_one_hot[:, int(last_agent)] = 1.0
            last_exists[:] = 1.0

        node_mean = node_probs.mean(dim=-1, keepdim=True)
        node_max = node_probs.max(dim=-1, keepdim=True).values
        edge_mean = edge_probs.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        edge_max = edge_probs.reshape(batch_size, -1).max(dim=-1, keepdim=True).values
        visited_ratio = visited_mask.mean(dim=-1, keepdim=True)
        revisit_ratio = (visits > 1).float().mean(dim=-1, keepdim=True)
        step_ratio = torch.full((batch_size, 1), float(step_idx) / max(1, max_steps), device=device)

        global_stats = torch.cat(
            [node_mean, node_max, edge_mean, edge_max, visited_ratio, revisit_ratio, step_ratio + 0.0 * last_exists],
            dim=-1,
        )
        return {
            "task_embedding": task_embedding,
            "node_probs": node_probs,
            "edge_probs": edge_probs,
            "visit_norm": visit_norm,
            "visited_mask": visited_mask,
            "last_one_hot": last_one_hot,
            "global_stats": global_stats,
        }

    def forward(self, state_repr: Dict[str, torch.Tensor], candidate_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if candidate_embeddings.dim() == 2:
            candidate_embeddings = candidate_embeddings.unsqueeze(0)

        task_embedding = state_repr["task_embedding"]
        node_probs = state_repr["node_probs"]
        edge_probs = state_repr["edge_probs"]
        visit_norm = state_repr["visit_norm"]
        visited_mask = state_repr["visited_mask"]
        last_one_hot = state_repr["last_one_hot"]
        global_stats = state_repr["global_stats"]

        global_input = torch.cat([task_embedding, global_stats], dim=-1)
        h = self.global_encoder(global_input)

        batch_size = candidate_embeddings.size(0)
        num_real_agents = candidate_embeddings.size(1)
        num_total_nodes = node_probs.size(-1)

        # [MOD] graph_prior 里应该比真实 agent 多 1 个 decision node
        if num_total_nodes != num_real_agents + 1:
            raise ValueError(
                f"graph_prior has {num_total_nodes} nodes, but candidate_embeddings has "
                f"{num_real_agents} real agents. Expected num_total_nodes = num_real_agents + 1."
            )

        # [MOD] 给 policy 的候选集拼上 decision node
        decision_embedding = self.decision_embedding.expand(batch_size, 1, -1)
        candidate_embeddings_ext = torch.cat([candidate_embeddings, decision_embedding], dim=1)

        out_mean = edge_probs.mean(dim=-1)
        in_mean = edge_probs.mean(dim=-2)
        candidate_stats = torch.stack(
            [node_probs, out_mean, in_mean, visit_norm, visited_mask, last_one_hot],
            dim=-1,
        )
        h_expand = h.unsqueeze(1).expand(-1, candidate_embeddings_ext.size(1), -1)
        candidate_input = torch.cat([h_expand, candidate_embeddings_ext, candidate_stats], dim=-1)
        logits = self.candidate_head(candidate_input).squeeze(-1)

        return {
            "next_agent_logits": logits,
            # "stop_logit": self.stop_head(h).squeeze(-1),
            "state_value": self.value_head(h).squeeze(-1),
        }
