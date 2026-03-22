# memory/memory_bank.py

import numpy as np
from typing import Dict, List


class MemoryBank:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.items: List[Dict] = []

    def add(self, item: Dict):
        # 往经验池里加新样本；如果池子没满就直接加，满了就只保留reward更高的版本。
        if len(self.items) < self.max_size:
            self.items.append(item)
            return

        min_idx = int(np.argmin([x["reward"] for x in self.items]))
        if item["reward"] > self.items[min_idx]["reward"]:
            self.items[min_idx] = item

    def retrieve(self, task_embedding: List[float], top_k: int = 1) -> List[Dict]:
        if len(self.items) == 0:
            return []

        query = np.array(task_embedding, dtype=np.float32)
        sims = []
        for item in self.items:
            emb = np.array(item["task_embedding"], dtype=np.float32)
            sim = float(np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8))
            sims.append(sim)

        sorted_idx = np.argsort(sims)[::-1][:top_k]
        return [self.items[i] for i in sorted_idx]