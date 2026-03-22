# env/reward.py

from typing import Dict


class RewardCalculator:
    def __init__(
        self,
        alpha_task: float = 1.0,
        beta_messages: float = 0.0,
        gamma_tokens: float = 0.001,
        delta_latency: float = 0.0,
    ):
        self.alpha_task = alpha_task
        self.beta_messages = beta_messages
        self.gamma_tokens = gamma_tokens
        self.delta_latency = delta_latency

    def compute(self, result: Dict) -> float:
        """
        第一版先简单一些：
        reward = task_score - token_cost_penalty - latency_penalty
        """
        reward = (
            self.alpha_task * float(result["task_score"])
            - self.gamma_tokens * float(result["total_tokens"])
            - self.delta_latency * float(result["latency"])
        )
        return reward