from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from GDesigner.llm.profile_embedding import get_sentence_embedding

from env.reward_v3 import V3RewardCalculator
from mas.executor import MultiAgentExecutor
from memory.trajectory_memory_bank import TrajectoryMemoryBank
from tasks.base_adapter import BaseTaskAdapter
from utils.v3_trace_utils import (
    build_trace_from_indices,
    build_task_agent_runtime,
    count_deadloops,
    trace_to_graph_spec,
    trace_to_local_indices,
    trace_to_semantic_edges,
)


class MultiAgentGraphV3Env:
    def __init__(
        self,
        tasks: List[Dict],
        task_adapter: BaseTaskAdapter,
        executor: MultiAgentExecutor,
        reward_calculator: V3RewardCalculator,
        memory_bank: TrajectoryMemoryBank,
        graph_generator=None,
        agent_profile_embeddings: Optional[torch.Tensor] = None,
        top_k_memory: int = 5,
        node_threshold: float = 0.35,
        edge_threshold: float = 0.35,
    ):
        self.tasks = tasks
        self.task_adapter = task_adapter
        self.executor = executor
        self.reward_calculator = reward_calculator
        self.memory_bank = memory_bank
        self.graph_generator = graph_generator
        self.agent_profile_embeddings = agent_profile_embeddings
        self.top_k_memory = top_k_memory
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold

        self.default_agent_profile_embeddings = agent_profile_embeddings
        self.current_task = None
        self.current_task_embedding = None
        self.current_summary = None
        self.current_graph_prior = None
        self.current_agent_names = list(executor.agent_names)
        self.current_node_kwargs = list(executor.node_kwargs or [{} for _ in executor.agent_names])
        self.current_agent_pool = []
        self.current_agent_profile_embeddings = agent_profile_embeddings

    @property
    def num_agents(self) -> int:
        if self.current_agent_pool:
            return len(self.current_agent_pool)
        return len(self.executor.agent_names)

    def encode_task(self, task_text: str) -> List[float]:
        return get_sentence_embedding(task_text).tolist()

    def _summary_to_tensors(self, summary: Dict) -> Dict[str, torch.Tensor]:
        return {
            "node_prior": torch.tensor(summary["node_prior"], dtype=torch.float32).unsqueeze(0),
            "edge_prior": torch.tensor(summary["edge_prior"], dtype=torch.float32).unsqueeze(0),
        }

    def reset(self) -> Dict:
        self.current_task = random.choice(self.tasks)
        task_text = self.task_adapter.get_task_text(self.current_task)
        self.current_task_embedding = self.encode_task(task_text)

        self.current_agent_names, self.current_node_kwargs, self.current_agent_pool, self.current_agent_profile_embeddings = build_task_agent_runtime(
            self.current_task,
            self.executor.agent_names,
            self.executor.node_kwargs,
        )

        self.current_summary = self.memory_bank.summarize(
            self.current_task_embedding,
            current_agent_pool=self.current_agent_pool,
            top_k=self.top_k_memory,
        )
        summary_tensors = self._summary_to_tensors(self.current_summary)
        task_tensor = torch.tensor(self.current_task_embedding, dtype=torch.float32).unsqueeze(0)

        if self.graph_generator is not None and self.current_agent_profile_embeddings is not None:
            with torch.no_grad():
                self.current_graph_prior = self.graph_generator.predict_graph(
                    task_tensor,
                    self.current_agent_profile_embeddings,
                    memory_summary=summary_tensors,
                )
        else:
            self.current_graph_prior = {
                "node_probs": summary_tensors["node_prior"],
                "edge_probs": summary_tensors["edge_prior"],
            }

        return {
            "task": self.current_task,
            "task_embedding": self.current_task_embedding,
            "memory_summary": self.current_summary,
            "graph_prior": self.current_graph_prior,
            "agent_pool": self.current_agent_pool,
            "candidate_agent_embeddings": self.current_agent_profile_embeddings,
            "agent_names": self.current_agent_names,
            "node_kwargs": self.current_node_kwargs,
        }

    async def execute_trace(self, trace):
        if len(trace) == 0:
            trace = build_trace_from_indices([0], self.current_agent_pool)

        result = await self.executor.run_trace(
            task=self.current_task,
            trace=trace,
            graph_prior=self.current_graph_prior,
            node_threshold=self.node_threshold,
            edge_threshold=self.edge_threshold,
            agent_names_override=self.current_agent_names,
            node_kwargs_override=self.current_node_kwargs,
        )
        local_trace = trace_to_local_indices(trace)
        route_stats = {"steps": len(local_trace), "deadloops": count_deadloops(trace, ngram=2)}
        reward = self.reward_calculator.compute(result, route_stats)
        info = {
            "result": result,
            "support_graph": trace_to_graph_spec(trace, num_agents=self.num_agents),
            "support_edges_semantic": trace_to_semantic_edges(trace),
            "execution_graph": result.get("graph", {}),
            "task_embedding": self.current_task_embedding,
            "task_text": self.current_task["task_text"],
            "trace": trace,
            "selected_trace": local_trace,
            "route_stats": route_stats,
            "agent_pool": self.current_agent_pool,
            "agent_names": self.current_agent_names,
            "node_kwargs": self.current_node_kwargs,
        }
        return reward, info
