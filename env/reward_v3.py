from __future__ import annotations

from typing import Dict


class V3RewardCalculator:
    def __init__(
        self,
        alpha_correctness: float = 1.0,
        beta_tokens: float = 0.001,
        gamma_steps: float = 0.05,
        delta_deadloop: float = 0.10,
        min_steps_free: int = 1, # [MOD] 第1步不罚，鼓励协作
        short_trace_penalty: float = 0.15, # [MOD] 错误且过短时额外惩罚
        short_trace_threshold: int = 1, # [MOD] 1 个 agent 就结束算“过短”
        use_binary_correctness: bool = True, # [MOD] 默认用 correct，而不是 task_score
    ):
        self.alpha_correctness = alpha_correctness
        self.beta_tokens = beta_tokens
        self.gamma_steps = gamma_steps
        self.delta_deadloop = delta_deadloop
        self.min_steps_free = min_steps_free
        self.short_trace_penalty = short_trace_penalty
        self.short_trace_threshold = short_trace_threshold
        self.use_binary_correctness = use_binary_correctness

    def compute(self, result: Dict, route_stats: Dict) -> float:
        task_score = float(result.get("task_score", 0.0))
        binary_correct = float(result.get("correct", 1.0 if task_score >= 1.0 else 0.0))
        
        correctness = binary_correct if self.use_binary_correctness else task_score

        token_cost = float(result["total_tokens"])
        steps = float(route_stats.get("steps", 0))
        deadloops = float(route_stats.get("deadloops", 0))

        # [MOD] 只对“正确样本”明显考虑效率，避免错误样本因为更短而占优
        token_penalty = self.beta_tokens * token_cost * correctness

        # [MOD] 前 min_steps_free 步不罚
        extra_steps = max(0.0, steps - float(self.min_steps_free))
        step_penalty = self.gamma_steps * extra_steps * correctness

        # [MOD] 对“错误且特别短”的轨迹额外打压，避免塌缩到单 agent 快速失败
        short_wrong_penalty = 0.0
        if correctness < 0.5 and steps <= float(self.short_trace_threshold):
            short_wrong_penalty = self.short_trace_penalty
            
        reward = (
            self.alpha_correctness * correctness
            - token_penalty
            - step_penalty
            - self.delta_deadloop * deadloops
            - short_wrong_penalty
        )
        return float(reward)
