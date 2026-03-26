from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.distributions import Bernoulli, Categorical

from utils.v3_trace_utils import trace_to_local_indices


class RouterSampler:
    def __init__(
        self,
        node_threshold: float = 0.35,
        edge_threshold: float = 0.35,
        force_non_empty_trace: bool = True,
    ):
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.force_non_empty_trace = force_non_empty_trace

    def build_valid_agent_mask(
        self,
        graph_prior: Dict[str, torch.Tensor],
        last_agent: Optional[int],
        trace,
    ) -> torch.Tensor:
        trace_indices = trace_to_local_indices(trace)
        node_probs = graph_prior["node_probs"]
        edge_probs = graph_prior["edge_probs"]
        if node_probs.dim() == 2:
            node_probs = node_probs[0]
        if edge_probs.dim() == 3:
            edge_probs = edge_probs[0]

        node_mask = (node_probs >= self.node_threshold).float()
        if last_agent is None:
            valid = node_mask.clone()
        else:
            valid = ((edge_probs[int(last_agent)] >= self.edge_threshold).float() * node_mask)

        if valid.sum() <= 0:
            valid = node_probs.clone()
            if last_agent is not None:
                valid[int(last_agent)] = 0.0

        if valid.sum() <= 0:
            valid = torch.ones_like(node_probs)
            if last_agent is not None:
                valid[int(last_agent)] = 0.0

        if self.force_non_empty_trace and len(trace_indices) == 0 and valid.sum() <= 0:
            valid = torch.ones_like(node_probs)

        return (valid > 0).float()

    def _build_distributions(self, policy_output: Dict[str, torch.Tensor], valid_agent_mask: torch.Tensor):
        if valid_agent_mask.dim() == 1:
            valid_agent_mask = valid_agent_mask.unsqueeze(0)

        stop_dist = Bernoulli(logits=policy_output["stop_logit"].unsqueeze(-1))
        masked_logits = policy_output["next_agent_logits"].clone()
        masked_logits = masked_logits.masked_fill(valid_agent_mask <= 0, -1e9)
        next_agent_dist = Categorical(logits=masked_logits)
        probs = torch.softmax(masked_logits, dim=-1)
        return stop_dist, next_agent_dist, masked_logits, probs

    def sample(
        self,
        policy_output: Dict[str, torch.Tensor],
        valid_agent_mask: torch.Tensor,
        trace,
    ) -> Dict:
        trace_indices = trace_to_local_indices(trace)
        stop_dist, next_agent_dist, masked_logits, probs = self._build_distributions(policy_output, valid_agent_mask)

        stop_sample = stop_dist.sample()
        use_stop_logprob = not (self.force_non_empty_trace and len(trace_indices) == 0)
        if not use_stop_logprob:
            stop_sample.zero_()

        next_agent = next_agent_dist.sample()
        use_agent_logprob = (int(stop_sample[0][0].item()) == 0) or (len(trace_indices) == 0)

        logprob = torch.zeros(1, device=masked_logits.device, dtype=masked_logits.dtype)
        entropy = torch.zeros(1, device=masked_logits.device, dtype=masked_logits.dtype)
        if use_agent_logprob:
            logprob = logprob + next_agent_dist.log_prob(next_agent)
            entropy = entropy + next_agent_dist.entropy()
        if use_stop_logprob:
            logprob = logprob + stop_dist.log_prob(stop_sample).squeeze(-1)
            entropy = entropy + stop_dist.entropy().squeeze(-1)

        picked = int(next_agent[0].item())

        action = {
            "stop": int(stop_sample[0][0].item()),
            "next_agent": picked,
            "selected_local_idx": picked,
            "selection_score": float(masked_logits[0, picked].item()),
            "selection_prob": float(probs[0, picked].item()),
            "allowed_by_graph": bool(valid_agent_mask[0, picked].item() > 0),
            "logprob": logprob,
            "entropy": entropy,
            "use_stop_logprob": use_stop_logprob,
            "use_agent_logprob": use_agent_logprob,
        }
        return action

    def evaluate_action(
        self,
        policy_output: Dict[str, torch.Tensor],
        valid_agent_mask: torch.Tensor,
        selected_local_idx: int,
        stop: int,
        use_stop_logprob: bool,
        use_agent_logprob: bool,
    ) -> Dict[str, torch.Tensor]:
        stop_dist, next_agent_dist, masked_logits, probs = self._build_distributions(policy_output, valid_agent_mask)

        selected = torch.tensor([int(selected_local_idx)], dtype=torch.long, device=masked_logits.device)
        stop_tensor = torch.tensor([[float(stop)]], dtype=torch.float32, device=masked_logits.device)

        logprob = torch.zeros(1, device=masked_logits.device, dtype=masked_logits.dtype)
        entropy = torch.zeros(1, device=masked_logits.device, dtype=masked_logits.dtype)
        if use_agent_logprob:
            logprob = logprob + next_agent_dist.log_prob(selected)
            entropy = entropy + next_agent_dist.entropy()
        if use_stop_logprob:
            logprob = logprob + stop_dist.log_prob(stop_tensor).squeeze(-1)
            entropy = entropy + stop_dist.entropy().squeeze(-1)

        return {
            "logprob": logprob,
            "entropy": entropy,
            "selection_prob": probs[0, int(selected_local_idx)],
        }
