from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from utils.v3_trace_utils import (
    count_deadloops,
    trace_to_graph_spec,
    trace_to_local_indices,
    trace_to_semantic_edges,
)


class TrajectoryMemoryBank:
    """
    V4 memory bank (rule-based V1)

    Core idea:
    - active positive bank: reusable high-quality traces
    - corrective bank: high predicted reward but low actual reward
    - prototype bank: distilled role/edge patterns updated by rules

    Backward-compatible fields kept:
    - self.items
    - add / retrieve / summarize / build_soft_labels / export_jsonl / load_jsonl
    """

    def __init__(
        self,
        max_size: int = 500,
        corrective_max_size: int = 200,
        prototype_match_threshold: float = 0.72,
        positive_reward_threshold: float = 0.0,
        corrective_gap_threshold: float = 0.35,
        corrective_low_reward_threshold: float = 0.0,
        negative_edge_penalty: float = 0.35,
    ):
        self.max_size = max_size
        self.corrective_max_size = corrective_max_size
        self.prototype_match_threshold = prototype_match_threshold
        self.positive_reward_threshold = positive_reward_threshold
        self.corrective_gap_threshold = corrective_gap_threshold
        self.corrective_low_reward_threshold = corrective_low_reward_threshold
        self.negative_edge_penalty = negative_edge_penalty

        # backward-compatible: positive active bank
        self.items: List[Dict[str, Any]] = []

        # new
        self.corrective_items: List[Dict[str, Any]] = [] # 模型原以为好，实际差的轨迹
        self.prototypes: List[Dict[str, Any]] = [] # 把多条轨迹压成的“模式原型”
        self.archived: List[Dict[str, Any]] = [] # 归档 / 淘汰 记录

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        if va.size == 0 or vb.size == 0:
            return 0.0
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        return float(np.dot(va, vb) / denom)

    @staticmethod
    def _edge_key(src: str, dst: str) -> str:
        return f"{src}|||{dst}"

    @staticmethod
    def _decode_edge_key(key: str) -> Tuple[str, str]:
        if "|||" not in key:
            return key, key
        src, dst = key.split("|||", 1)
        return src, dst

    @staticmethod
    def _pythonize(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist") and not isinstance(value, list):
            value = value.tolist()
        return deepcopy(value)

    # ------------------------------------------------------------------
    # trace abstraction
    # ------------------------------------------------------------------
    def _trace_to_role_sequence(
        self,
        trace: Sequence[Any],
        agent_pool: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        roles: List[str] = []
        for step in trace or []:
            if isinstance(step, dict):
                role = (
                    step.get("agent_role")
                    or step.get("role")
                    or step.get("agent_name")
                    or "UnknownRole"
                )
                roles.append(str(role))
                continue

            idx = int(step)
            if agent_pool is not None and 0 <= idx < len(agent_pool):
                role = (
                    agent_pool[idx].get("agent_role")
                    or agent_pool[idx].get("role")
                    or agent_pool[idx].get("agent_name")
                    or f"Agent{idx}"
                )
                roles.append(str(role))
            else:
                roles.append(f"Agent{idx}")
        return roles

    def _role_edges(self, role_seq: Sequence[str]) -> List[Tuple[str, str]]:
        edges = []
        for i in range(len(role_seq) - 1):
            edges.append((str(role_seq[i]), str(role_seq[i + 1])))
        return edges

    def _role_edge_counter(self, role_seq: Sequence[str]) -> Dict[str, float]:
        counter = Counter()
        for src, dst in self._role_edges(role_seq):
            counter[self._edge_key(src, dst)] += 1.0
        return dict(counter)

    def _sequence_similarity(self, a: Sequence[str], b: Sequence[str]) -> float:
        if not a or not b:
            return 0.0
        m = min(len(a), len(b))
        same = sum(1 for i in range(m) if a[i] == b[i])
        return same / max(len(a), len(b))

    def _edge_jaccard(self, edge_a: Dict[str, float], edge_b: Dict[str, float]) -> float:
        ka = set(edge_a.keys())
        kb = set(edge_b.keys())
        if not ka and not kb:
            return 1.0
        if not ka or not kb:
            return 0.0
        return len(ka & kb) / max(1, len(ka | kb))

    def _infer_agent_set_id(self, item: Dict[str, Any]) -> str:
        if item.get("agent_set_id"):
            return str(item["agent_set_id"])
        pool = item.get("agent_pool") or []
        if pool:
            names = [str(x.get("agent_name", f"Agent{i}")) for i, x in enumerate(pool)]
            return "|".join(names)
        return "default"

    def _compute_quality_score(self, item: Dict[str, Any]) -> float:
        reward = self._safe_float(item.get("actual_reward", item.get("reward", 0.0)))
        correct = 1.0 if int(item.get("correct", 0)) > 0 else 0.0
        loops = self._safe_float(item.get("loop_count", 0.0))
        cost = self._safe_float(item.get("token_cost", 0.0))
        steps = self._safe_float(item.get("steps", 0.0))

        # very simple rule-based score
        return (
            1.50 * correct
            + 1.00 * reward
            - 0.15 * loops
            - 0.00005 * cost
            - 0.03 * max(0.0, steps - 2.0)
        )

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(item)
        trace = item.get("trace")
        if trace is None:
            trace = item.get("selected_trace", [])
        item["trace"] = trace

        item.setdefault("exp_id", uuid4().hex[:12])
        item.setdefault("selected_trace", trace_to_local_indices(trace))

        route_stats = dict(item.get("route_stats") or {})
        item.setdefault("reward", self._safe_float(item.get("reward", 0.0)))
        item.setdefault("actual_reward", self._safe_float(item.get("actual_reward", item["reward"])))
        item.setdefault("predicted_reward", self._safe_float(item.get("predicted_reward", 0.0)))
        item.setdefault("reward_gap", self._safe_float(item["predicted_reward"]) - self._safe_float(item["actual_reward"]))

        item.setdefault("correct", int(item.get("correct", 0)))
        item.setdefault("steps", len(item.get("selected_trace", [])))
        item.setdefault("token_cost", self._safe_float(item.get("token_cost", item.get("total_tokens", 0.0))))

        num_agents = len(item.get("agent_pool", [])) or int(item.get("num_agents", 0))
        if num_agents <= 0 and item.get("selected_trace"):
            num_agents = max(item["selected_trace"]) + 1
        item.setdefault("num_agents", num_agents)

        loop_count = item.get("loop_count")
        if loop_count is None:
            loop_count = route_stats.get("deadloops")
        if loop_count is None:
            loop_count = count_deadloops(item["selected_trace"], ngram=2)
        item["loop_count"] = int(loop_count)

        item.setdefault("support_graph", trace_to_graph_spec(item.get("trace", []), num_agents))
        item.setdefault("support_edges_semantic", trace_to_semantic_edges(item.get("trace", [])))
        item.setdefault("agent_pool", [])
        item.setdefault("task_embedding", self._pythonize(item.get("task_embedding")) or [])
        item.setdefault("task_text", "")
        item.setdefault("task_family", item.get("task_type", "generic"))
        item.setdefault("agent_set_id", self._infer_agent_set_id(item))
        item.setdefault("status", "active")

        abstract_roles = self._trace_to_role_sequence(item["trace"], item["agent_pool"])
        item["abstract_roles"] = abstract_roles
        item["abstract_edges"] = self._role_edge_counter(abstract_roles)
        item["stop_depth"] = int(len(item.get("selected_trace", [])))
        item["quality_score"] = self._safe_float(item.get("quality_score", self._compute_quality_score(item)))
        item.setdefault("writeback_type", str(item.get("writeback_type", "")) or "unknown")
        item.setdefault("writeback_reason", str(item.get("writeback_reason", "")))

        raw_steps = item.get("steps_detail") or []
        normalized_steps = []
        for s in raw_steps:
            if not isinstance(s, dict):
                continue
            normalized_steps.append(
                {
                    "t": int(s.get("t", len(normalized_steps))),
                    "role": str(s.get("role", "")),
                    "agent": str(s.get("agent", "")),
                    "selected_packets": list(s.get("selected_packets", []) or []),
                    "output_packet_types": list(s.get("output_packet_types", []) or []),
                }
            )
        item["steps_detail"] = normalized_steps

        return item

    # ------------------------------------------------------------------
    # writeback rules
    # ------------------------------------------------------------------
    def _decide_writeback_type(self, item: Dict[str, Any]) -> Tuple[str, str]:
        actual = self._safe_float(item.get("actual_reward", item.get("reward", 0.0)))
        pred = self._safe_float(item.get("predicted_reward", 0.0))
        gap = pred - actual
        correct = int(item.get("correct", 0))

        if gap >= self.corrective_gap_threshold and actual <= self.corrective_low_reward_threshold:
            return "corrective", "overestimated_failure"

        if actual >= self.positive_reward_threshold or correct > 0:
            return "positive", "high_actual_reward"

        return "none", "below_writeback_threshold"

    def _best_positive_proto_match(self, item: Dict[str, Any]) -> Tuple[Optional[int], float]:
        if not self.prototypes:
            return None, 0.0

        best_idx = None
        best_score = -1.0
        for idx, proto in enumerate(self.prototypes):
            if proto.get("status", "active") != "active":
                continue
            if proto.get("task_family") != item.get("task_family"):
                continue

            task_sim = self._cosine(
                item.get("task_embedding", []),
                proto.get("centroid_task_embedding", []),
            )
            role_sim = self._sequence_similarity(
                item.get("abstract_roles", []),
                proto.get("prototype_roles", []),
            )
            edge_sim = self._edge_jaccard(
                item.get("abstract_edges", {}),
                proto.get("edge_weight_pos", {}),
            )
            sim = 0.45 * task_sim + 0.25 * role_sim + 0.30 * edge_sim
            if sim > best_score:
                best_score = sim
                best_idx = idx
        return best_idx, max(0.0, best_score)

    def _update_proto_stats(self, proto: Dict[str, Any], item: Dict[str, Any], positive: bool) -> None:
        reward = self._safe_float(item.get("actual_reward", 0.0))
        cost = self._safe_float(item.get("token_cost", 0.0))
        steps = self._safe_float(item.get("steps", 0.0))
        quality = max(0.05, self._safe_float(item.get("quality_score", 0.0)))

        proto.setdefault("support_exp_ids", [])
        proto["support_exp_ids"].append(item["exp_id"])
        proto["support_count"] = int(proto.get("support_count", 0)) + (1 if positive else 0)
        proto["corrective_count"] = int(proto.get("corrective_count", 0)) + (0 if positive else 1)
        proto["total_seen"] = int(proto.get("total_seen", 0)) + 1
        proto["avg_reward"] = (
            (self._safe_float(proto.get("avg_reward", 0.0)) * max(0, proto["total_seen"] - 1) + reward)
            / max(1, proto["total_seen"])
        )
        proto["avg_cost"] = (
            (self._safe_float(proto.get("avg_cost", 0.0)) * max(0, proto["total_seen"] - 1) + cost)
            / max(1, proto["total_seen"])
        )
        proto["avg_steps"] = (
            (self._safe_float(proto.get("avg_steps", 0.0)) * max(0, proto["total_seen"] - 1) + steps)
            / max(1, proto["total_seen"])
        )
        proto["success_rate"] = (
            (self._safe_float(proto.get("success_rate", 0.0)) * max(0, proto["total_seen"] - 1) + float(int(item.get("correct", 0))))
            / max(1, proto["total_seen"])
        )

        # centroid task embedding
        emb = item.get("task_embedding", [])
        if emb:
            if not proto.get("centroid_task_embedding"):
                proto["centroid_task_embedding"] = deepcopy(emb)
            else:
                old = np.array(proto["centroid_task_embedding"], dtype=np.float32)
                new = np.array(emb, dtype=np.float32)
                proto["centroid_task_embedding"] = (
                    0.8 * old + 0.2 * new
                ).tolist()

        # prototype roles
        if not proto.get("prototype_roles"):
            proto["prototype_roles"] = list(item.get("abstract_roles", []))

        # stop-depth histogram
        stop_hist = dict(proto.get("stop_depth_hist") or {})
        stop_key = str(int(item.get("stop_depth", len(item.get("selected_trace", [])))))
        stop_hist[stop_key] = int(stop_hist.get(stop_key, 0)) + 1
        proto["stop_depth_hist"] = stop_hist

        # positive / negative edge accumulators
        pos_map = dict(proto.get("edge_weight_pos") or {})
        neg_map = dict(proto.get("edge_weight_neg") or {})
        source_edges = dict(item.get("abstract_edges") or {})

        if positive:
            for k, v in source_edges.items():
                pos_map[k] = self._safe_float(pos_map.get(k, 0.0)) + quality * self._safe_float(v, 0.0)
        else:
            penalty = max(0.10, self._safe_float(item.get("reward_gap", 0.0)))
            for k, v in source_edges.items():
                neg_map[k] = self._safe_float(neg_map.get(k, 0.0)) + penalty * self._safe_float(v, 0.0)

        proto["edge_weight_pos"] = pos_map
        proto["edge_weight_neg"] = neg_map
        proto["version"] = int(proto.get("version", 0)) + 1

    def _create_prototype(self, item: Dict[str, Any]) -> Dict[str, Any]:
        proto = {
            "proto_id": f"proto_{uuid4().hex[:10]}",
            "task_family": item.get("task_family", "generic"),
            "agent_set_scope": item.get("agent_set_id", "default"),
            "centroid_task_embedding": deepcopy(item.get("task_embedding", [])),
            "prototype_roles": list(item.get("abstract_roles", [])),
            "edge_weight_pos": {},
            "edge_weight_neg": {},
            "stop_depth_hist": {},
            "support_exp_ids": [],
            "support_count": 0,
            "corrective_count": 0,
            "total_seen": 0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_cost": 0.0,
            "avg_steps": 0.0,
            "version": 0,
            "status": "active",
        }
        self._update_proto_stats(proto, item, positive=True)
        return proto

    def _archive_record(self, record_type: str, source_id: str, reason: str) -> None:
        self.archived.append(
            {
                "record_id": f"arc_{uuid4().hex[:10]}",
                "record_type": record_type,
                "source_id": source_id,
                "archive_reason": reason,
            }
        )

    def _evict_lowest_quality_positive(self) -> None:
        if len(self.items) <= self.max_size:
            return
        min_idx = int(np.argmin([self._safe_float(x.get("quality_score", 0.0)) for x in self.items]))
        removed = self.items.pop(min_idx)
        self._archive_record("episodic_positive", removed.get("exp_id", "unknown"), "positive_bank_overflow")

    def _evict_lowest_value_corrective(self) -> None:
        if len(self.corrective_items) <= self.corrective_max_size:
            return
        min_idx = int(np.argmin([self._safe_float(x.get("reward_gap", 0.0)) for x in self.corrective_items]))
        removed = self.corrective_items.pop(min_idx)
        self._archive_record("episodic_corrective", removed.get("exp_id", "unknown"), "corrective_bank_overflow")

    def add(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item = self._normalize_item(item)

        writeback_type = item.get("writeback_type")
        writeback_reason = item.get("writeback_reason", "")
        if writeback_type in (None, "", "unknown"):
            writeback_type, writeback_reason = self._decide_writeback_type(item)
            item["writeback_type"] = writeback_type
            item["writeback_reason"] = writeback_reason

        if writeback_type == "none":
            return {
                "op": "NOOP",
                "target_id": None,
                "reason": writeback_reason,
                "item_id": item["exp_id"],
            }

        if writeback_type == "corrective":
            self.corrective_items.append(item)
            self._evict_lowest_value_corrective() # 经验池可能有容量，把价值最低的corrective item淘汰

            proto_idx, proto_sim = self._best_positive_proto_match(item)
            target_id = None
            if proto_idx is not None and proto_sim >= self.prototype_match_threshold:
                self._update_proto_stats(self.prototypes[proto_idx], item, positive=False)
                target_id = self.prototypes[proto_idx]["proto_id"]

            return {
                "op": "ADD_CORRECTIVE",
                "target_id": target_id,
                "reason": writeback_reason,
                "item_id": item["exp_id"],
            }

        # positive
        proto_idx, proto_sim = self._best_positive_proto_match(item)
        item["novelty_score"] = 1.0 - proto_sim

        self.items.append(item)
        self._evict_lowest_quality_positive()

        if proto_idx is not None and proto_sim >= self.prototype_match_threshold:
            self._update_proto_stats(self.prototypes[proto_idx], item, positive=True)
            return {
                "op": "UPDATE",
                "target_id": self.prototypes[proto_idx]["proto_id"],
                "reason": f"{writeback_reason}|proto_sim={proto_sim:.3f}",
                "item_id": item["exp_id"],
            }

        new_proto = self._create_prototype(item)
        self.prototypes.append(new_proto)
        return {
            "op": "ADD",
            "target_id": new_proto["proto_id"],
            "reason": f"{writeback_reason}|new_pattern",
            "item_id": item["exp_id"],
        }

    def add_many(self, items: List[Dict[str, Any]]) -> None:
        for item in items:
            self.add(item)

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def _retrieval_score(self, item: Dict[str, Any], task_embedding: Sequence[float]) -> float:
        task_sim = self._cosine(task_embedding, item.get("task_embedding", []))
        quality = self._safe_float(item.get("quality_score", 0.0))
        correct_bonus = 0.15 * float(int(item.get("correct", 0)))
        loop_penalty = 0.06 * self._safe_float(item.get("loop_count", 0.0))
        cost_penalty = 0.00002 * self._safe_float(item.get("token_cost", 0.0))
        return task_sim + 0.25 * quality + correct_bonus - loop_penalty - cost_penalty

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
            if min_reward is not None and self._safe_float(item.get("actual_reward", item.get("reward", 0.0))) < min_reward:
                continue
            score = self._retrieval_score(item, task_embedding)
            enriched = dict(item)
            enriched["_retrieval_score"] = float(score)
            scored.append((score, idx, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, _, item in scored[:top_k]]

    def retrieve_corrective(
        self,
        task_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not self.corrective_items:
            return []

        scored = []
        for idx, item in enumerate(self.corrective_items):
            task_sim = self._cosine(task_embedding, item.get("task_embedding", []))
            gap = self._safe_float(item.get("reward_gap", 0.0))
            score = task_sim + 0.35 * gap
            enriched = dict(item)
            enriched["_retrieval_score"] = float(score)
            scored.append((score, idx, enriched))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    # mapping retrieved traces/prototypes to current runtime agent set
    # ------------------------------------------------------------------
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
            weight = max(0.05, self._safe_float(item.get("_retrieval_score", item.get("quality_score", 0.0))))
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
                    node_prior[mapped_idx] += weight * max(0.10, sim)

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
                score = weight * max(0.10, src_sim) * max(0.10, dst_sim)
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
            "avg_reward": float(np.mean([self._safe_float(x.get("actual_reward", x.get("reward", 0.0))) for x in retrieved])) if retrieved else 0.0,
            "avg_correct": float(np.mean([int(x.get("correct", 0)) for x in retrieved])) if retrieved else 0.0,
            "avg_steps": float(np.mean([self._safe_float(x.get("steps", 0.0)) for x in retrieved])) if retrieved else 0.0,
        }

    def _summarize_prototypes(
        self,
        task_embedding: List[float],
        current_agent_pool: Sequence[Dict[str, Any]],
        top_k: int = 2,
    ) -> Dict[str, Any]:
        num_agents = len(current_agent_pool)
        if num_agents <= 0 or not self.prototypes:
            return {
                "node_prior": [0.0 for _ in range(num_agents)],
                "edge_prior": [[0.0 for _ in range(num_agents)] for _ in range(num_agents)],
            }

        scored = []
        for proto in self.prototypes:
            if proto.get("status", "active") != "active":
                continue
            sim = self._cosine(task_embedding, proto.get("centroid_task_embedding", []))
            score = sim + 0.20 * self._safe_float(proto.get("success_rate", 0.0))
            scored.append((score, proto))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [p for _, p in scored[:top_k]]

        node_prior = [0.0 for _ in range(num_agents)]
        edge_prior = [[0.0 for _ in range(num_agents)] for _ in range(num_agents)]

        role_to_idx = {}
        for idx, agent in enumerate(current_agent_pool):
            role = str(agent.get("agent_role", "") or agent.get("role", "") or agent.get("agent_name", ""))
            role_to_idx[role] = idx

        total_w = 0.0
        for proto in picked:
            w = max(0.05, self._safe_float(proto.get("success_rate", 0.0)) + 0.20 * self._safe_float(proto.get("avg_reward", 0.0)))
            total_w += w

            for role in proto.get("prototype_roles", []):
                if role in role_to_idx:
                    node_prior[role_to_idx[role]] += w

            pos_map = dict(proto.get("edge_weight_pos") or {})
            neg_map = dict(proto.get("edge_weight_neg") or {})
            all_keys = set(pos_map.keys()) | set(neg_map.keys())
            for k in all_keys:
                src_role, dst_role = self._decode_edge_key(k)
                if src_role not in role_to_idx or dst_role not in role_to_idx:
                    continue
                src_idx = role_to_idx[src_role]
                dst_idx = role_to_idx[dst_role]
                score = self._safe_float(pos_map.get(k, 0.0)) - self.negative_edge_penalty * self._safe_float(neg_map.get(k, 0.0))
                edge_prior[src_idx][dst_idx] += w * max(0.0, score)

        if total_w <= 0:
            return {
                "node_prior": [0.0 for _ in range(num_agents)],
                "edge_prior": [[0.0 for _ in range(num_agents)] for _ in range(num_agents)],
            }

        node_prior = [min(1.0, max(0.0, x / total_w)) for x in node_prior]
        edge_prior = [[min(1.0, max(0.0, x / total_w)) for x in row] for row in edge_prior]
        return {
            "node_prior": node_prior,
            "edge_prior": edge_prior,
        }

    def _build_routing_memory(
        self,
        positive_summary: Dict[str, Any],
        corrective_summary: Dict[str, Any],
        current_agent_pool: Sequence[Dict[str, Any]],
        retrieved: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        names = [
            str(x.get("agent_role") or x.get("role") or x.get("agent_name") or f"Agent{i}")
            for i, x in enumerate(current_agent_pool)
        ]

        def top_edges(mat: List[List[float]], top_n: int = 8) -> List[Dict[str, Any]]:
            arr = []
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if i == j:
                        continue
                    arr.append((float(mat[i][j]), i, j))
            arr.sort(key=lambda x: x[0], reverse=True)
            out = []
            for score, i, j in arr[:top_n]:
                if score <= 0:
                    continue
                out.append({"src": names[i], "dst": names[j], "score": float(score)})
            return out

        prefix_counter: Dict[str, Counter] = defaultdict(Counter)
        loop_counter: Counter = Counter()
        stop_depths: List[int] = []
        risky_prefix_counter: Counter = Counter()

        for item in retrieved:
            roles = list(item.get("abstract_roles", []))
            stop_depths.append(int(item.get("stop_depth", len(item.get("selected_trace", [])))))
            for i in range(len(roles) - 1):
                prefix = "|".join(roles[: i + 1])
                prefix_counter[prefix][roles[i + 1]] += 1
            idxs = item.get("selected_trace", [])
            for i in range(len(idxs) - 2):
                gram = tuple(idxs[i : i + 3])
                if len(set(gram)) < len(gram):
                    loop_counter[str(gram)] += 1
        
        corrective_retrieved = corrective_summary.get("_retrieved_items", []) if isinstance(corrective_summary, dict) else []
        for item in corrective_retrieved:
            roles = list(item.get("abstract_roles", []))
            for i in range(1, len(roles)):
                prefix = "|".join(roles[: i + 1])
                risky_prefix_counter[prefix] += 1

        prefix_next_step = {}
        for prefix, counter in prefix_counter.items():
            total = sum(counter.values())
            if total <= 0:
                continue
            prefix_next_step[prefix] = {
                k: float(v / total) for k, v in counter.most_common(3)
            }

        return {
            "recommended_edges": top_edges(positive_summary["edge_prior"]),
            "risky_edges": top_edges(corrective_summary["edge_prior"]),
            "prefix_next_step": prefix_next_step,
            "avg_stop_depth": float(np.mean(stop_depths)) if stop_depths else 0.0,
            "common_loops": [{"pattern": k, "count": int(v)} for k, v in loop_counter.most_common(5)],
            "risky_prefixes": [
                {"prefix": k, "count": int(v)}
                for k, v in risky_prefix_counter.most_common(8)
            ],
        }

    def _build_execution_memory(
        self,
        retrieved: Sequence[Dict[str, Any]],
        current_agent_pool: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current_roles = [
            str(x.get("agent_role") or x.get("role") or x.get("agent_name") or "UnknownRole")
            for x in current_agent_pool
        ]
        role_stats: Dict[str, Dict[str, Any]] = {}
        role_packet_priors: Dict[str, Counter] = {}
        role_output_packet_priors: Dict[str, Counter] = {}
        for role in current_roles:
            role_packet_priors[role] = Counter()
            role_output_packet_priors[role] = Counter()

        for role in current_roles:
            role_stats[role] = {
                "times_seen": 0,
                "avg_position": 0.0,
                "common_prev_roles": Counter(),
                "common_next_roles": Counter(),
            }

        for item in retrieved:
            roles = list(item.get("abstract_roles", []))
            for step in item.get("steps_detail", []) or []:
                role = str(step.get("role", "") or "")
                if role not in role_packet_priors:
                    continue
                for pkt in step.get("selected_packets", []) or []:
                    role_packet_priors[role][str(pkt)] += 1
                for pkt in step.get("output_packet_types", []) or []:
                    role_output_packet_priors[role][str(pkt)] += 1

            for i, role in enumerate(roles):
                if role not in role_stats:
                    continue
                role_stats[role]["times_seen"] += 1
                role_stats[role]["avg_position"] += float(i)
                if i > 0:
                    role_stats[role]["common_prev_roles"][roles[i - 1]] += 1
                if i + 1 < len(roles):
                    role_stats[role]["common_next_roles"][roles[i + 1]] += 1

        formatted = {}
        for role, stat in role_stats.items():
            seen = max(1, int(stat["times_seen"]))
            formatted[role] = {
                "times_seen": int(stat["times_seen"]),
                "avg_position": float(stat["avg_position"] / seen) if stat["times_seen"] > 0 else -1.0,
                "common_prev_roles": dict(stat["common_prev_roles"].most_common(3)),
                "common_next_roles": dict(stat["common_next_roles"].most_common(3)),
            }

        trace_examples = []
        for item in list(retrieved)[:3]:
            trace_examples.append(
                {
                    "roles": list(item.get("abstract_roles", [])),
                    "reward": float(item.get("actual_reward", item.get("reward", 0.0))),
                    "correct": int(item.get("correct", 0)),
                }
            )

        formatted_packet_priors = {}
        formatted_output_packet_priors = {}
        for role in current_roles:
            in_counter = role_packet_priors.get(role, Counter())
            out_counter = role_output_packet_priors.get(role, Counter())

            in_total = sum(in_counter.values())
            out_total = sum(out_counter.values())

            formatted_packet_priors[role] = (
                {k: float(v / max(1, in_total)) for k, v in in_counter.most_common(6)}
                if in_total > 0 else {}
            )
            formatted_output_packet_priors[role] = (
                {k: float(v / max(1, out_total)) for k, v in out_counter.most_common(6)}
                if out_total > 0 else {}
            )

        return {
            "role_hints": formatted,
            "role_packet_priors": formatted_packet_priors,
            "role_output_packet_priors": formatted_output_packet_priors,
            "trace_examples": trace_examples,
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
        positive_summary = self._summarize_retrieved(retrieved, current_agent_pool=current_agent_pool)
        corrective_retrieved = self.retrieve_corrective(task_embedding, top_k=max(1, top_k // 2))
        corrective_summary = self._summarize_retrieved(corrective_retrieved, current_agent_pool=current_agent_pool)
        corrective_summary["_retrieved_items"] = list(corrective_retrieved)
        proto_summary = self._summarize_prototypes(task_embedding, current_agent_pool=current_agent_pool, top_k=2)

        num_agents = len(current_agent_pool)
        if num_agents <= 0:
            num_agents = 0

        node_prior = []
        edge_prior = []
        for i in range(num_agents):
            node_score = (
                float(positive_summary["node_prior"][i])
                + 0.25 * float(proto_summary["node_prior"][i])
                - self.negative_edge_penalty * float(corrective_summary["node_prior"][i])
            )
            node_prior.append(min(1.0, max(0.0, node_score)))

        for i in range(num_agents):
            row = []
            for j in range(num_agents):
                edge_score = (
                    float(positive_summary["edge_prior"][i][j])
                    + 0.25 * float(proto_summary["edge_prior"][i][j])
                    - self.negative_edge_penalty * float(corrective_summary["edge_prior"][i][j])
                )
                row.append(min(1.0, max(0.0, edge_score)))
            edge_prior.append(row)

        if not retrieved and num_agents > 0:
            node_prior = [0.5 for _ in range(num_agents)]
            edge_prior = [[0.5 for _ in range(num_agents)] for _ in range(num_agents)]

        return {
            "retrieved": list(retrieved),
            "retrieved_corrective": list(corrective_retrieved),
            "node_prior": node_prior,
            "edge_prior": edge_prior,
            "avg_reward": positive_summary["avg_reward"],
            "avg_correct": positive_summary["avg_correct"],
            "avg_steps": positive_summary["avg_steps"],
            "routing_memory": self._build_routing_memory(
                positive_summary=positive_summary,
                corrective_summary=corrective_summary,
                current_agent_pool=current_agent_pool,
                retrieved=retrieved,
            ),
            "execution_memory": self._build_execution_memory(
                retrieved=retrieved,
                current_agent_pool=current_agent_pool,
            ),
            "bank_stats": {
                "positive_size": len(self.items),
                "corrective_size": len(self.corrective_items),
                "prototype_size": len(self.prototypes),
                "archived_size": len(self.archived),
            },
        }

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
            fallback_summary = self._summarize_retrieved(
                [self._normalize_item(fallback_item)],
                current_agent_pool=current_agent_pool,
            )
            return {
                "node_targets": fallback_summary["node_prior"],
                "edge_targets": fallback_summary["edge_prior"],
            }

        return {
            "node_targets": [0.5 for _ in range(num_agents)],
            "edge_targets": [[0.5 for _ in range(num_agents)] for _ in range(num_agents)],
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def export_jsonl(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with path_obj.open("w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps({"record_kind": "positive", **item}, ensure_ascii=False) + "\n")
            for item in self.corrective_items:
                f.write(json.dumps({"record_kind": "corrective", **item}, ensure_ascii=False) + "\n")
            for proto in self.prototypes:
                f.write(json.dumps({"record_kind": "prototype", **proto}, ensure_ascii=False) + "\n")
            for arc in self.archived:
                f.write(json.dumps({"record_kind": "archived", **arc}, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            return

        self.items = []
        self.corrective_items = []
        self.prototypes = []
        self.archived = []

        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                kind = raw.pop("record_kind", "positive")

                if kind == "positive":
                    self.items.append(self._normalize_item(raw))
                elif kind == "corrective":
                    self.corrective_items.append(self._normalize_item(raw))
                elif kind == "prototype":
                    self.prototypes.append(raw)
                elif kind == "archived":
                    self.archived.append(raw)