from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.v3_trace_utils import trace_to_graph_spec, trace_to_local_indices, trace_to_semantic_edges


class TrajectoryMemoryBank:
    """Experience bank for V3.

    Runtime still uses local indices to execute the current episode, but experience is
    stored in a *semantic* form (`trace`, `agent_pool`, `support_edges_semantic`) so
    the memory can be reused across tasks or across different agent sets.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.items: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.items)

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(item)
        trace = item.get("trace")
        if trace is None:
            trace = item.get("selected_trace", [])
        item["trace"] = trace
        item.setdefault("selected_trace", trace_to_local_indices(trace))
        item.setdefault("reward", 0.0)
        item.setdefault("correct", 0)
        item.setdefault("steps", len(item.get("selected_trace", [])))
        item.setdefault("token_cost", item.get("total_tokens", 0.0))
        num_agents = len(item.get("agent_pool", [])) or int(item.get("num_agents", 0))
        if num_agents <= 0 and item.get("selected_trace"):
            num_agents = max(item["selected_trace"]) + 1
        item.setdefault("num_agents", num_agents)
        item.setdefault("support_graph", trace_to_graph_spec(item.get("trace", []), num_agents))
        item.setdefault("support_edges_semantic", trace_to_semantic_edges(item.get("trace", [])))
        item.setdefault("agent_pool", [])
        return item

    def add(self, item: Dict[str, Any]) -> None:
        item = self._normalize_item(item)

        if len(self.items) < self.max_size:
            self.items.append(item)
            return

        scores = [float(x.get("reward", 0.0)) for x in self.items]
        min_idx = int(np.argmin(scores))
        if float(item["reward"]) > scores[min_idx]:
            self.items[min_idx] = item

    def add_many(self, items: List[Dict[str, Any]]) -> None:
        for item in items:
            self.add(item)

    def _cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        return float(np.dot(va, vb) / denom)

    def retrieve(
        self,
        task_embedding: List[float],
        top_k: int = 5,
        min_reward: Optional[float] = None,
        exclude_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.items:
            return []

        scored = []
        for idx, item in enumerate(self.items):
            if exclude_index is not None and idx == exclude_index:
                continue
            if min_reward is not None and float(item.get("reward", 0.0)) < min_reward:
                continue
            sim = self._cosine(task_embedding, item["task_embedding"])
            scored.append((sim, idx, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, _, item in scored[:top_k]]

    def _current_embeddings(self, current_agent_pool: Sequence[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        embs: List[Optional[np.ndarray]] = []
        for agent in current_agent_pool:
            emb = agent.get("agent_embedding")
            if emb is None:
                embs.append(None)
            else:
                embs.append(np.array(emb, dtype=np.float32))
        return embs

    def _best_match(
        self,
        embedding: Optional[Sequence[float]],
        role: Optional[str],
        name: Optional[str],
        current_agent_pool: Sequence[Dict[str, Any]],
        current_embeddings: Sequence[Optional[np.ndarray]],
    ) -> Tuple[Optional[int], float]:
        if embedding is not None:
            src = np.array(embedding, dtype=np.float32)
            best_idx = None
            best_sim = -1.0
            for idx, cand in enumerate(current_embeddings):
                if cand is None:
                    continue
                denom = float(np.linalg.norm(src) * np.linalg.norm(cand) + 1e-8)
                sim = float(np.dot(src, cand) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            if best_idx is not None:
                return best_idx, max(0.0, best_sim)

        role = (role or "").lower().strip()
        name = (name or "").lower().strip()
        for idx, agent in enumerate(current_agent_pool):
            cand_role = str(agent.get("agent_role", "")).lower().strip()
            cand_name = str(agent.get("agent_name", "")).lower().strip()
            if role and role == cand_role:
                return idx, 1.0
            if name and name == cand_name:
                return idx, 1.0
        return None, 0.0

    def _summarize_retrieved(
        self,
        retrieved: Sequence[Dict[str, Any]],
        current_agent_pool: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        num_agents = len(current_agent_pool)
        if num_agents <= 0:
            return {
                "node_prior": [],
                "edge_prior": [],
                "avg_reward": 0.0,
                "avg_correct": 0.0,
                "avg_steps": 0.0,
            }

        node_prior = [0.0 for _ in range(num_agents)]
        edge_prior = [[0.0 for _ in range(num_agents)] for _ in range(num_agents)]
        current_embeddings = self._current_embeddings(current_agent_pool)
        total_w = 0.0

        for item in retrieved:
            weight = max(0.0, float(item.get("reward", 0.0))) + 1e-3
            total_w += weight
            trace = item.get("trace", [])
            semantic_edges = item.get("support_edges_semantic") or trace_to_semantic_edges(trace)

            for step in trace:
                if not isinstance(step, dict):
                    idx = int(step)
                    if 0 <= idx < num_agents:
                        node_prior[idx] += weight
                    continue
                mapped_idx, sim = self._best_match(
                    embedding=step.get("agent_embedding"),
                    role=step.get("agent_role"),
                    name=step.get("agent_name"),
                    current_agent_pool=current_agent_pool,
                    current_embeddings=current_embeddings,
                )
                if mapped_idx is not None:
                    node_prior[mapped_idx] += weight * max(0.1, sim)

            for edge in semantic_edges:
                src_idx, src_sim = self._best_match(
                    embedding=edge.get("src_embedding"),
                    role=edge.get("src_role"),
                    name=edge.get("src_name"),
                    current_agent_pool=current_agent_pool,
                    current_embeddings=current_embeddings,
                )
                dst_idx, dst_sim = self._best_match(
                    embedding=edge.get("dst_embedding"),
                    role=edge.get("dst_role"),
                    name=edge.get("dst_name"),
                    current_agent_pool=current_agent_pool,
                    current_embeddings=current_embeddings,
                )
                if src_idx is None or dst_idx is None or src_idx == dst_idx:
                    continue
                score = weight * max(0.1, src_sim) * max(0.1, dst_sim)
                edge_prior[src_idx][dst_idx] += score
                node_prior[src_idx] += 0.5 * score
                node_prior[dst_idx] += 0.5 * score

        if total_w <= 0:
            return {
                "node_prior": [0.5 for _ in range(num_agents)],
                "edge_prior": [[0.5 for _ in range(num_agents)] for _ in range(num_agents)],
                "avg_reward": 0.0,
                "avg_correct": 0.0,
                "avg_steps": 0.0,
            }

        node_prior = [min(1.0, max(0.0, x / total_w)) for x in node_prior]
        edge_prior = [[min(1.0, max(0.0, x / total_w)) for x in row] for row in edge_prior]
        return {
            "node_prior": node_prior,
            "edge_prior": edge_prior,
            "avg_reward": float(np.mean([x.get("reward", 0.0) for x in retrieved])) if retrieved else 0.0,
            "avg_correct": float(np.mean([x.get("correct", 0.0) for x in retrieved])) if retrieved else 0.0,
            "avg_steps": float(np.mean([x.get("steps", 0.0) for x in retrieved])) if retrieved else 0.0,
        }

    def summarize(
        self,
        task_embedding: List[float],
        current_agent_pool: Sequence[Dict[str, Any]],
        top_k: int = 5,
        min_reward: Optional[float] = None,
        exclude_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        retrieved = self.retrieve(
            task_embedding,
            top_k=top_k,
            min_reward=min_reward,
            exclude_index=exclude_index,
        )
        if not retrieved:
            num_agents = len(current_agent_pool)
            return {
                "retrieved": [],
                "node_prior": [0.5 for _ in range(num_agents)],
                "edge_prior": [[0.5 for _ in range(num_agents)] for _ in range(num_agents)],
                "avg_reward": 0.0,
                "avg_correct": 0.0,
                "avg_steps": 0.0,
            }

        summary = self._summarize_retrieved(retrieved, current_agent_pool=current_agent_pool)
        summary["retrieved"] = list(retrieved)
        return summary

    def build_soft_labels(
        self,
        task_embedding: List[float],
        current_agent_pool: Sequence[Dict[str, Any]],
        top_k: int = 5,
        fallback_item: Optional[Dict[str, Any]] = None,
        exclude_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        summary = self.summarize(
            task_embedding,
            current_agent_pool=current_agent_pool,
            top_k=top_k,
            exclude_index=exclude_index,
        )
        if summary["retrieved"]:
            return {
                "node_targets": summary["node_prior"],
                "edge_targets": summary["edge_prior"],
            }

        num_agents = len(current_agent_pool)
        if fallback_item is not None:
            fallback_summary = self._summarize_retrieved([self._normalize_item(fallback_item)], current_agent_pool=current_agent_pool)
            return {
                "node_targets": fallback_summary["node_prior"],
                "edge_targets": fallback_summary["edge_prior"],
            }

        return {
            "node_targets": [0.5 for _ in range(num_agents)],
            "edge_targets": [[0.5 for _ in range(num_agents)] for _ in range(num_agents)],
        }

    def export_jsonl(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            return
        self.items = []
        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(self._normalize_item(json.loads(line)))
