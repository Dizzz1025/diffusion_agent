# policy/graph_sampler.py

import torch
from torch.distributions import Bernoulli


class GraphSampler:
    def sample(self, policy_output):
        use_memory_dist = Bernoulli(logits=policy_output["use_memory_logit"])
        node_dist = Bernoulli(logits=policy_output["node_logits"])
        edge_dist = Bernoulli(logits=policy_output["edge_logits"])

        use_memory = use_memory_dist.sample()
        node_mask = node_dist.sample()
        edge_mask = edge_dist.sample()

        logprob = (
            use_memory_dist.log_prob(use_memory).sum(dim=-1)
            + node_dist.log_prob(node_mask).sum(dim=-1)
            + edge_dist.log_prob(edge_mask).sum(dim=[1, 2])
        )

        action = {
            "use_memory": int(use_memory[0][0].item()),
            "node_mask": node_mask[0].int().tolist(),
            "edge_mask": edge_mask[0].int().tolist(),
        }
        return action, logprob