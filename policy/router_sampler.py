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

        decision_idx = self._decision_idx(node_probs)        # [MOD]
        real_node_probs = node_probs[:decision_idx]          # [MOD] 前 N 个是真实 agent
        valid = torch.zeros_like(node_probs)                 # [MOD]

        real_node_mask = (real_node_probs >= self.node_threshold).float() # node_mask.shape = [4]
        if last_agent is None:
            # [MOD] 第一步只能选真实 agent，不能直接选 decision node
            valid[:decision_idx] = real_node_mask
            valid[decision_idx] = 0.0
        else:
            valid[:decision_idx] = (
                (edge_probs[int(last_agent), :decision_idx] >= self.edge_threshold).float()
                * real_node_mask
            )

            # [MOD] decision node 是否可选，也由图边来控制
            valid[decision_idx] = float(
                edge_probs[int(last_agent), decision_idx].item() >= self.edge_threshold
            )
        
        # 原逻辑保留：如果严格图约束下没有真实 agent 可选，则回退到更宽松的 real-agent 候选
        if valid[:decision_idx].sum() <= 0:
            valid[:decision_idx] = real_node_probs.clone()
            if last_agent is not None and 0 <= int(last_agent) < decision_idx:
                valid[int(last_agent)] = 0.0

        if valid[:decision_idx].sum() <= 0:
            valid[:decision_idx] = torch.ones_like(real_node_probs)
            if last_agent is not None and 0 <= int(last_agent) < decision_idx:
                valid[int(last_agent)] = 0.0

        if self.force_non_empty_trace and len(trace_indices) == 0:
            valid[decision_idx] = 0.0

        return (valid > 0).float()

    def _build_distributions(self, policy_output: Dict[str, torch.Tensor], valid_agent_mask: torch.Tensor):
        if valid_agent_mask.dim() == 1:
            valid_agent_mask = valid_agent_mask.unsqueeze(0)

        # stop_dist = Bernoulli(logits=policy_output["stop_logit"].unsqueeze(-1))
        masked_logits = policy_output["next_agent_logits"].clone()
        masked_logits = masked_logits.masked_fill(valid_agent_mask <= 0, -1e9)
        next_agent_dist = Categorical(logits=masked_logits)
        probs = torch.softmax(masked_logits, dim=-1)
        return next_agent_dist, masked_logits, probs

    def sample(
        self,
        policy_output: Dict[str, torch.Tensor],
        valid_agent_mask: torch.Tensor,
        trace,
    ) -> Dict:
        next_agent_dist, masked_logits, probs = self._build_distributions(policy_output, valid_agent_mask)

        next_agent = next_agent_dist.sample()
        picked = int(next_agent[0].item())

        decision_idx = self._decision_idx(valid_agent_mask)
        is_decision = (picked == decision_idx)

        logprob = next_agent_dist.log_prob(next_agent)
        entropy = next_agent_dist.entropy()

        allowed = (
            bool(valid_agent_mask[0, picked].item() > 0)
            if valid_agent_mask.dim() == 2
            else bool(valid_agent_mask[picked].item() > 0)
        )


        action = {
            "stop": int(is_decision),
            "next_agent": picked,
            "selected_local_idx": picked,
            "selection_score": float(masked_logits[0, picked].item()),
            "selection_prob": float(probs[0, picked].item()),
            "allowed_by_graph": allowed,
            "logprob": logprob,
            "entropy": entropy,
            # [MOD] 兼容字段，已经不再真的使用 stop 的 logprob
            "use_stop_logprob": False,
            "use_agent_logprob": True,
            # [MOD] 新字段：明确告诉 rollout 这一步是不是 decision node
            "selected_is_decision": bool(is_decision),
        }
        return action

    def evaluate_action(
        self,
        policy_output: Dict[str, torch.Tensor],
        valid_agent_mask: torch.Tensor,
        selected_local_idx: int,
        stop: int,                 # [MOD] 兼容旧 trainer 签名，实际不再使用
        use_stop_logprob: bool,    # [MOD] 兼容旧 trainer 签名，实际不再使用
        use_agent_logprob: bool,   # [MOD] 兼容旧 trainer 签名，实际不再使用
    ) -> Dict[str, torch.Tensor]:
        # stop_dist, next_agent_dist, masked_logits, probs = self._build_distributions(policy_output, valid_agent_mask)
        next_agent_dist, masked_logits, probs = self._build_distributions(
            policy_output,
            valid_agent_mask,
        )
        selected = torch.tensor([int(selected_local_idx)], dtype=torch.long, device=masked_logits.device)
        # stop_tensor = torch.tensor([[float(stop)]], dtype=torch.float32, device=masked_logits.device)

        logprob = next_agent_dist.log_prob(selected)
        entropy = next_agent_dist.entropy()

        return {
            "logprob": logprob,
            "entropy": entropy,
            "selection_prob": probs[0, int(selected_local_idx)],
        }
    
    def _decision_idx(self, x: torch.Tensor) -> int:
        return int(x.size(-1) - 1)  # [MOD] 最后一个节点固定视为 decision node
