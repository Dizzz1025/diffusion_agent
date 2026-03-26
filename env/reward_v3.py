from __future__ import annotations

from typing import Dict


class V3RewardCalculator:
    def __init__(
        self,
        alpha_correctness: float = 1.0,
        beta_tokens: float = 0.001,
        gamma_steps: float = 0.05,
        delta_deadloop: float = 0.10,
    ):
        self.alpha_correctness = alpha_correctness
        self.beta_tokens = beta_tokens
        self.gamma_steps = gamma_steps
        self.delta_deadloop = delta_deadloop

    def compute(self, result: Dict, route_stats: Dict) -> float:
        correctness = float(result["task_score"])
        token_cost = float(result["total_tokens"])
        steps = float(route_stats.get("steps", 0))
        deadloops = float(route_stats.get("deadloops", 0))
        reward = (
            self.alpha_correctness * correctness
            - self.beta_tokens * token_cost
            - self.gamma_steps * steps
            - self.delta_deadloop * deadloops
        )
        return float(reward)
