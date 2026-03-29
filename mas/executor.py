from __future__ import annotations

import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens
from tasks.base_adapter import BaseTaskAdapter
from utils.v3_trace_utils import build_agent_pool, trace_to_local_indices


class MultiAgentExecutor:
    """Executor for both fixed-graph execution and trace-driven execution."""

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

    def build_agent_pool_snapshot(
        self,
        agent_embeddings: Optional[Any] = None,
        agent_names_override: Optional[List[str]] = None,
        node_kwargs_override: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        agent_names = agent_names_override if agent_names_override is not None else self.agent_names
        node_kwargs = node_kwargs_override if node_kwargs_override is not None else self.node_kwargs
        return build_agent_pool(agent_names, node_kwargs, agent_embeddings=agent_embeddings)

    def _build_graph(
        self,
        spatial_mask: Optional[List[List[int]]] = None,
        temporal_mask: Optional[List[List[int]]] = None,
        agent_names_override: Optional[List[str]] = None,
        node_kwargs_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Graph:
        agent_names = agent_names_override if agent_names_override is not None else self.agent_names
        node_kwargs = node_kwargs_override if node_kwargs_override is not None else self.node_kwargs
        return Graph(
            domain=self.domain,
            llm_name=self.llm_name,
            agent_names=agent_names,
            decision_method=self.decision_method,
            optimized_spatial=self.optimized_spatial,
            optimized_temporal=self.optimized_temporal,
            fixed_spatial_masks=spatial_mask,
            fixed_temporal_masks=temporal_mask,
            node_kwargs=node_kwargs,
        )

    @staticmethod
    def _tensor_or_list_to_python(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            while x.dim() > 0 and x.size(0) == 1:
                x = x.squeeze(0)
            return x.tolist()
        return x

    def _extract_trace_constraints(
        self,
        graph_prior: Optional[Dict[str, Any]],
        node_threshold: float,
        edge_threshold: float,
    ) -> Tuple[Optional[List[float]], Optional[List[List[float]]], Optional[List[int]], Optional[List[List[int]]]]:
        if graph_prior is None:
            return None, None, None, None

        node_probs = self._tensor_or_list_to_python(graph_prior.get("node_probs"))
        edge_probs = self._tensor_or_list_to_python(graph_prior.get("edge_probs"))
        node_mask = self._tensor_or_list_to_python(graph_prior.get("node_mask"))
        edge_mask = self._tensor_or_list_to_python(graph_prior.get("edge_mask"))

        if node_mask is None and node_probs is not None:
            node_mask = [1 if float(v) >= node_threshold else 0 for v in node_probs]
        if edge_mask is None and edge_probs is not None:
            edge_mask = [[1 if float(v) >= edge_threshold else 0 for v in row] for row in edge_probs]
        return node_probs, edge_probs, node_mask, edge_mask

    def _node_ids_in_order(self, graph: Graph, expected_num_agents: Optional[int] = None) -> List[str]:
        node_ids = list(graph.nodes.keys())
        target_num = len(self.agent_names) if expected_num_agents is None else int(expected_num_agents)
        if len(node_ids) != target_num:
            raise ValueError(
                f"Graph initialized {len(node_ids)} nodes, but executor expects {target_num} agents."
            )
        return node_ids

    @staticmethod
    def _snapshot_counters() -> Tuple[float, int, int]:
        return (Cost.instance().value, PromptTokens.instance().value, CompletionTokens.instance().value)

    @staticmethod
    def _counter_delta(before: Tuple[float, int, int], after: Tuple[float, int, int]) -> Dict[str, float]:
        cost_before, pt_before, ct_before = before
        cost_after, pt_after, ct_after = after
        return {
            "cost": cost_after - cost_before,
            "prompt_tokens": pt_after - pt_before,
            "completion_tokens": ct_after - ct_before,
            "total_tokens": (pt_after - pt_before) + (ct_after - ct_before),
        }

    def _format_result(
        self,
        task: Dict[str, Any],
        raw_answer: Any,
        log_prob: float,
        start_ts: float,
        counter_before: Tuple[float, int, int],
        graph_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        eval_result = self.task_adapter.evaluate_answer(raw_answer, task)
        deltas = self._counter_delta(counter_before, self._snapshot_counters())
        return {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "task_text": task["task_text"],
            "ground_truth": task["ground_truth"],
            "raw_answer": raw_answer,
            "predict_answer": eval_result["predict_answer"],
            "correct": eval_result["correct"],
            "task_score": eval_result["task_score"],
            "log_prob": log_prob,
            **deltas,
            "latency": time.time() - start_ts,
            "graph": graph_payload,
        }

    async def run(
        self,
        task: Dict[str, Any],
        graph_spec: Dict[str, Any],
        agent_names_override: Optional[List[str]] = None,
        node_kwargs_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        start_ts = time.time()
        spatial_mask = graph_spec.get("spatial_mask", None)
        temporal_mask = graph_spec.get("temporal_mask", None)
        counter_before = self._snapshot_counters()

        graph = self._build_graph(
            spatial_mask=spatial_mask,
            temporal_mask=temporal_mask,
            agent_names_override=agent_names_override,
            node_kwargs_override=node_kwargs_override,
        )
        input_dict = self.task_adapter.build_input_dict(task)
        raw_answer, log_prob = await graph.arun(input_dict, self.num_rounds)

        return self._format_result(
            task=task,
            raw_answer=raw_answer,
            log_prob=log_prob,
            start_ts=start_ts,
            counter_before=counter_before,
            graph_payload={**graph_spec, "agent_names": agent_names_override or self.agent_names},
        )

    async def run_trace(
        self,
        task: Dict[str, Any],
        trace,
        graph_prior: Optional[Dict[str, Any]] = None,
        node_threshold: float = 0.35,
        edge_threshold: float = 0.35,
        allow_all_previous_if_disconnected: bool = True,
        agent_names_override: Optional[List[str]] = None,
        node_kwargs_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute a router-produced trace directly.

        The generated graph constrains which agents can be selected by the router, but
        the *execution context* itself follows the trace prefix: at step ``t`` the
        current node sees the outputs from all previous trace steps.
        """
        del allow_all_previous_if_disconnected  # kept only for backward-compatible signature
        start_ts = time.time()
        counter_before = self._snapshot_counters()
        input_dict = self.task_adapter.build_input_dict(task)

        active_agent_names = agent_names_override if agent_names_override is not None else self.agent_names
        graph = self._build_graph(
            agent_names_override=agent_names_override,
            node_kwargs_override=node_kwargs_override,
        )
        node_ids = self._node_ids_in_order(graph, expected_num_agents=len(active_agent_names))
        graph.clear_spatial_connection()
        graph.clear_temporal_connection()

        _, _, node_mask, edge_mask = self._extract_trace_constraints(
            graph_prior=graph_prior,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        local_trace = trace_to_local_indices(trace)
        safe_trace: List[int] = []
        for agent_idx in local_trace:
            idx = int(agent_idx)
            if 0 <= idx < len(node_ids):
                if node_mask is not None and int(node_mask[idx]) <= 0:
                    continue
                safe_trace.append(idx)
        if not safe_trace:
            safe_trace = [0]

        executed_step_snapshots: List[SimpleNamespace] = []
        execution_records = []

        for step_idx, agent_idx in enumerate(safe_trace):
            node_id = node_ids[agent_idx]
            current_node = graph.nodes[node_id]

            prefix_predecessors = [snap for snap in executed_step_snapshots if getattr(snap, "outputs", None)]
            current_node.spatial_predecessors = prefix_predecessors
            current_node.temporal_predecessors = []

            await current_node.async_execute(input_dict)
            current_node.update_memory()

            snapshot = SimpleNamespace(
                id=f"{node_id}#step{step_idx}",
                node_id=node_id,
                local_idx=agent_idx,
                role=getattr(current_node, "role", None),
                outputs=deepcopy(current_node.outputs),
            )
            executed_step_snapshots.append(snapshot)
            execution_records.append(
                {
                    "step": step_idx,
                    "agent_idx": agent_idx,
                    "node_id": node_id,
                    "prefix_predecessors": [pred.id for pred in prefix_predecessors],
                    "revisit": sum(1 for x in safe_trace[: step_idx + 1] if x == agent_idx) > 1,
                }
            )

        graph.decision_node.spatial_predecessors = list(executed_step_snapshots)
        graph.decision_node.temporal_predecessors = []
        await graph.decision_node.async_execute(input_dict)
        raw_answer = graph.decision_node.outputs
        if len(raw_answer) == 0:
            raw_answer = ["No answer of the decision node"]

        result = self._format_result(
            task=task,
            raw_answer=raw_answer,
            log_prob=0.0,
            start_ts=start_ts,
            counter_before=counter_before,
            graph_payload={
                "execution_mode": "trace_direct_prefix_context",
                "selected_trace": list(safe_trace),
                "graph_prior": {"node_mask": node_mask, "edge_mask": edge_mask},
                "agent_names": list(active_agent_names),
                "execution_records": execution_records,
            },
        )
        return result
