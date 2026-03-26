from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGenerator(nn.Module):
    """Weakly supervised graph generator for V3.

    Input:
      - task embedding
      - agent profile embeddings
      - retrieved memory summary (optional)
    Output:
      - node activation probabilities
      - edge existence probabilities
    """

    def __init__(
        self,
        task_dim: int,
        agent_dim: int,
        hidden_dim: int,
        num_agents: int,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim + num_agents + num_agents * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim + agent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * agent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        task_embedding: torch.Tensor,
        agent_embeddings: torch.Tensor,
        memory_summary: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)
        if agent_embeddings.dim() == 2:
            agent_embeddings = agent_embeddings.unsqueeze(0)

        batch_size = task_embedding.size(0)
        num_agents = agent_embeddings.size(1)
        device = task_embedding.device

        if memory_summary is None:
            node_prior = torch.full((batch_size, num_agents), 0.5, device=device)
            edge_prior = torch.full((batch_size, num_agents, num_agents), 0.5, device=device)
        else:
            node_prior = memory_summary["node_prior"].to(device)
            edge_prior = memory_summary["edge_prior"].to(device)

        task_features = torch.cat(
            [
                task_embedding,
                node_prior,
                edge_prior.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        context = self.task_encoder(task_features)

        context_nodes = context.unsqueeze(1).expand(-1, num_agents, -1)
        node_inputs = torch.cat([context_nodes, agent_embeddings], dim=-1)
        node_logits = self.node_head(node_inputs).squeeze(-1)

        src_emb = agent_embeddings.unsqueeze(2).expand(-1, num_agents, num_agents, -1)
        dst_emb = agent_embeddings.unsqueeze(1).expand(-1, num_agents, num_agents, -1)
        context_edges = context.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, num_agents, -1)
        edge_inputs = torch.cat([context_edges, src_emb, dst_emb, edge_prior.unsqueeze(-1)], dim=-1)
        edge_logits = self.edge_head(edge_inputs).squeeze(-1)

        eye = torch.eye(num_agents, device=device).unsqueeze(0)
        edge_logits = edge_logits * (1.0 - eye) - 1e4 * eye

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "node_probs": torch.sigmoid(node_logits),
            "edge_probs": torch.sigmoid(edge_logits),
        }

    @torch.no_grad()
    def predict_graph(
        self,
        task_embedding: torch.Tensor,
        agent_embeddings: torch.Tensor,
        memory_summary: Optional[Dict[str, torch.Tensor]] = None,
        node_threshold: float = 0.45,
        edge_threshold: float = 0.45,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(task_embedding, agent_embeddings, memory_summary=memory_summary)
        node_mask = (outputs["node_probs"] >= node_threshold).float()
        edge_mask = (outputs["edge_probs"] >= edge_threshold).float()
        return {**outputs, "node_mask": node_mask, "edge_mask": edge_mask}


def graph_generator_loss(outputs: Dict[str, torch.Tensor], node_targets: torch.Tensor, edge_targets: torch.Tensor) -> torch.Tensor:
    node_loss = F.binary_cross_entropy_with_logits(outputs["node_logits"], node_targets)
    edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], edge_targets)
    return node_loss + edge_loss
