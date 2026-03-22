# mas/executor.py

import time
from typing import Any, Dict, List, Optional

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens

from tasks.base_adapter import BaseTaskAdapter


class MultiAgentExecutor:
    """
    负责：
    1. 根据 graph spec 构造 GDesigner Graph
    2. 执行一次多智能体协作
    3. 收集原始回答、log_prob、token/cost 等统计
    4. 调用 task adapter 做任务评测
    """

    def __init__(
        self,
        domain: str,
        llm_name: str,
        agent_names: List[str],
        decision_method: str,
        num_rounds: int,
        task_adapter: BaseTaskAdapter,
        optimized_spatial: bool = False,
        optimized_temporal: bool = False,
        node_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.domain = domain
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.decision_method = decision_method
        self.num_rounds = num_rounds
        self.task_adapter = task_adapter
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.node_kwargs = node_kwargs

    async def run(
        self,
        task: Dict[str, Any],
        graph_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        graph_spec 示例：
        {
            "spatial_mask": [[...], [...], ...],
            "temporal_mask": [[...], [...], ...] or None
        }
        """
        start_ts = time.time()

        spatial_mask = graph_spec.get("spatial_mask", None)
        temporal_mask = graph_spec.get("temporal_mask", None)

        # 记录执行前的全局统计，便于得到“本次 episode 增量”
        cost_before = Cost.instance().value
        pt_before = PromptTokens.instance().value
        ct_before = CompletionTokens.instance().value

        graph = Graph(
            domain=self.domain,
            llm_name=self.llm_name,
            agent_names=self.agent_names,
            decision_method=self.decision_method,
            optimized_spatial=self.optimized_spatial,
            optimized_temporal=self.optimized_temporal,
            fixed_spatial_masks=spatial_mask,
            fixed_temporal_masks=temporal_mask,
            node_kwargs=self.node_kwargs,
        )

        input_dict = self.task_adapter.build_input_dict(task)
        raw_answer, log_prob = await graph.arun(input_dict, self.num_rounds) # 如果不是从概率图中采样出图结构，log_prob会返回0

        # 任务评估交给 adapter
        eval_result = self.task_adapter.evaluate_answer(raw_answer, task)

        cost_after = Cost.instance().value
        pt_after = PromptTokens.instance().value
        ct_after = CompletionTokens.instance().value

        result = {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "task_text": task["task_text"],
            "ground_truth": task["ground_truth"],
            "raw_answer": raw_answer,
            "predict_answer": eval_result["predict_answer"],
            "correct": eval_result["correct"],
            "task_score": eval_result["task_score"],
            "log_prob": log_prob,
            "cost": cost_after - cost_before,
            "prompt_tokens": pt_after - pt_before,
            "completion_tokens": ct_after - ct_before,
            "total_tokens": (pt_after - pt_before) + (ct_after - ct_before),
            "latency": time.time() - start_ts,
            "graph": graph_spec,
        }
        return result