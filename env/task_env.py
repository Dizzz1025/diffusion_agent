# env/task_env.py

import random
import numpy as np
from typing import Dict, List, Optional

from memory.memory_bank import MemoryBank
from diffusion_agent.mas.executor_GTD import MultiAgentExecutor
from env.reward import RewardCalculator
from tasks.base_adapter import BaseTaskAdapter

from GDesigner.llm.profile_embedding import get_sentence_embedding

class MultiAgentGraphEnv:
    def __init__(
        self,
        tasks: List[Dict],
        task_adapter: BaseTaskAdapter,
        executor: MultiAgentExecutor,
        reward_calculator: RewardCalculator,
        memory_bank: Optional[MemoryBank] = None,
    ):
        self.tasks = tasks
        self.task_adapter = task_adapter
        self.executor = executor
        self.reward_calculator = reward_calculator
        self.memory_bank = memory_bank

        self.current_task = None
        self.current_task_embedding = None
        self.current_retrieved = []

    def encode_task(self, task_text: str) -> List[float]:
        return get_sentence_embedding(task_text).tolist()

    def reset(self) -> Dict:
        self.current_task = random.choice(self.tasks)
        task_text = self.task_adapter.get_task_text(self.current_task)
        self.current_task_embedding = self.encode_task(task_text)

        retrieved = []
        if self.memory_bank is not None:
            retrieved = self.memory_bank.retrieve(self.current_task_embedding, top_k=1)

        self.current_retrieved = retrieved

        state = {
            "task": self.current_task,
            "task_embedding": self.current_task_embedding,
            "retrieved": retrieved,
        }
        return state

    async def step(self, action: Dict):
        """
        action 示例：
        {
            "use_memory": 0/1,
            "node_mask": [0,1,1,1],
            "edge_mask": [[...], [...], ...]
        }
        """
        graph_spec = self.build_graph_from_action(action)
        result = await self.executor.run(self.current_task, graph_spec)
        reward = self.reward_calculator.compute(result)

        done = True
        next_state = None
        info = {
            "result": result,
            "graph_spec": graph_spec,
            "task_embedding": self.current_task_embedding,
        }
        return next_state, reward, done, info

    def build_graph_from_action(self, action: Dict) -> Dict:
        use_memory = int(action.get("use_memory", 0))
        node_mask = action["node_mask"]
        edge_mask = action["edge_mask"]

        # 至少保留一个节点
        if sum(node_mask) == 0:
            node_mask[0] = 1

        # 先把 edge_mask 和 node_mask 对齐
        N = len(node_mask)
        for i in range(N):
            for j in range(N):
                if i == j:
                    edge_mask[i][j] = 0
                if node_mask[i] == 0 or node_mask[j] == 0:
                    edge_mask[i][j] = 0

        # 如果使用 memory，就和 prior graph 简单融合
        if use_memory == 1 and len(self.current_retrieved) > 0:
            prior_graph = self.current_retrieved[0]["graph"]
            prior_edge_mask = prior_graph["spatial_mask"]
            edge_mask = self.merge_graphs(prior_edge_mask, edge_mask)

        return {
            "spatial_mask": edge_mask,
            "temporal_mask": None
        }

    def merge_graphs(self, prior_edge_mask, new_edge_mask):
        N = len(new_edge_mask)
        merged = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                merged[i][j] = 1 if (prior_edge_mask[i][j] == 1 or new_edge_mask[i][j] == 1) else 0
        return merged