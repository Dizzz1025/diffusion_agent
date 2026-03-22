# policy/graph_policy.py

import torch
import torch.nn as nn

class GraphPolicy(nn.Module):
    def __init__(self, state_dim=128, hidden_dim=256, num_agents=4):
        super().__init__()
        self.num_agents = num_agents

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 是否使用经验图
        self.memory_gate = nn.Linear(hidden_dim, 1)

        # 哪些节点激活
        self.node_head = nn.Linear(hidden_dim, num_agents)

        # 哪些边保留
        self.edge_head = nn.Linear(hidden_dim, num_agents * num_agents)

    def forward(self, state_vec):
        h = self.backbone(state_vec)

        use_memory_logit = self.memory_gate(h)                  # [B, 1]
        node_logits = self.node_head(h)                         # [B, N]
        edge_logits = self.edge_head(h).view(-1, self.num_agents, self.num_agents)  # [B, N, N]

        return {
            "use_memory_logit": use_memory_logit,
            "node_logits": node_logits,
            "edge_logits": edge_logits
        }