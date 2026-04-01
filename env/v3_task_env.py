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
        task_sampling_mode="random", # "random" / "sequential"
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
        self.task_sampling_mode = task_sampling_mode
        self.current_task_idx = None

        self.default_agent_profile_embeddings = agent_profile_embeddings
        self.current_task = None
        self.current_task_embedding = None
        self.current_summary = None
        self.current_graph_prior = None
        self.current_execution_graph_prior = None  # [MOD] 新增：给 executor 用的真实图
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
    
    def _extend_graph_prior_with_decision(
        self,
        graph_prior: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        [MOD]
        给 router 用的图先验：
        - 前 N 个节点是真实 agent
        - 最后 1 个节点是 decision node
        - 所有真实 agent -> decision node 的边先设成 1.0
        这样最小改动就能实现“选到 decision node 就结束”
        """
        node_probs = graph_prior["node_probs"]
        edge_probs = graph_prior["edge_probs"]

        if node_probs.dim() == 1:
            node_probs = node_probs.unsqueeze(0)
        if edge_probs.dim() == 2:
            edge_probs = edge_probs.unsqueeze(0)

        batch_size, num_real_agents = node_probs.shape
        node_dtype = node_probs.dtype
        edge_dtype = edge_probs.dtype
        node_device = node_probs.device
        edge_device = edge_probs.device

        # [MOD] 给 decision node 一个固定可用的 node prior
        node_probs_ext = torch.cat(
            [
                node_probs,
                torch.ones(batch_size, 1, dtype=node_dtype, device=node_device),
            ],
            dim=-1,
        )

        # [MOD] 扩边矩阵
        edge_probs_ext = torch.zeros(
            batch_size,
            num_real_agents + 1,
            num_real_agents + 1,
            dtype=edge_dtype,
            device=edge_device,
        )
        edge_probs_ext[:, :num_real_agents, :num_real_agents] = edge_probs

        # [MOD] 所有真实 agent 都允许转到 decision node
        edge_probs_ext[:, :num_real_agents, num_real_agents] = 1.0

        # [MOD] decision 自环可留着，纯粹为了矩阵完整
        edge_probs_ext[:, num_real_agents, num_real_agents] = 1.0

        return {
            "node_probs": node_probs_ext,
            "edge_probs": edge_probs_ext,
        }
    
    def reset(self) -> Dict:
        if self.task_sampling_mode == "random":
            self.current_task_idx = random.randrange(len(self.tasks))
            self.current_task = self.tasks[self.current_task_idx]
        elif self.task_sampling_mode == "sequential":
            if self.current_task_idx is None:
                self.current_task_idx = 0
            else:
                self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
            self.current_task = self.tasks[self.current_task_idx]
        else:
            raise ValueError(f"Unsupported task_sampling_mode: {self.task_sampling_mode}")
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

        # [MOD] 先拿“真实 agent 图”，给 executor 用
        if self.graph_generator is not None and self.current_agent_profile_embeddings is not None:
            with torch.no_grad():
                self.current_execution_graph_prior = self.graph_generator.predict_graph(
                    task_tensor,
                    self.current_agent_profile_embeddings,
                    memory_summary=summary_tensors,
                )
        else:
            self.current_execution_graph_prior = {
                "node_probs": summary_tensors["node_prior"],
                "edge_probs": summary_tensors["edge_prior"],
            }

        # [MOD] 再基于真实图扩成“router 图 = 真实 agent + decision node”
        self.current_graph_prior = self._extend_graph_prior_with_decision(
            self.current_execution_graph_prior
        )

        return {
            "task": self.current_task,
            "task_id": self.current_task_idx,
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
            graph_prior=self.current_execution_graph_prior,
            node_threshold=self.node_threshold,
            edge_threshold=self.edge_threshold,
            memory_summary=self.current_summary,
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
