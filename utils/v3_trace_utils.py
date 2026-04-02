from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from GDesigner.llm.profile_embedding import get_sentence_embedding


TraceStep = Dict[str, Any]
TraceLike = Sequence[Union[int, TraceStep]]


def _to_python_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    return deepcopy(value)




def resolve_task_agent_config(
    task: Optional[Dict[str, Any]],
    default_agent_names: Sequence[str],
    default_node_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Resolve the agent pool config carried by the current task.

    Supported task formats:
    - task["agent_pool"] = [{agent_name/agent_role/agent_desc/...}, ...]
    - task["agent_names"] + optional task["node_kwargs"]
    - otherwise fall back to the executor defaults
    """
    default_names = list(default_agent_names)
    default_kwargs = [dict(x) if isinstance(x, dict) else {} for x in (default_node_kwargs or [{} for _ in default_names])]

    if not task:
        return default_names, default_kwargs

    if isinstance(task.get("agent_pool"), list) and task.get("agent_pool"):
        pool = task["agent_pool"]
        agent_names: List[str] = []
        node_kwargs: List[Dict[str, Any]] = []
        for idx, item in enumerate(pool):
            item = dict(item) if isinstance(item, dict) else {}
            agent_names.append(item.get("agent_name", default_names[idx] if idx < len(default_names) else f"Agent{idx}"))
            node_kwargs.append(
                {
                    "role": item.get("agent_role", item.get("role", agent_names[-1])),
                    "desc": item.get("agent_desc", item.get("desc", item.get("description", item.get("agent_role", agent_names[-1])))),
                }
            )
        return agent_names, node_kwargs

    if isinstance(task.get("agent_names"), list) and task.get("agent_names"):
        agent_names = [str(x) for x in task["agent_names"]]
        raw_kwargs = task.get("node_kwargs") or [{} for _ in agent_names]
        node_kwargs = [dict(x) if isinstance(x, dict) else {} for x in raw_kwargs]
        while len(node_kwargs) < len(agent_names):
            node_kwargs.append({})
        return agent_names, node_kwargs

    return default_names, default_kwargs


def build_agent_profile_embeddings_from_pool(agent_pool: Sequence[Dict[str, Any]]) -> torch.Tensor:
    texts = []
    for item in agent_pool:
        texts.append(item.get("agent_desc") or item.get("agent_role") or item.get("agent_name") or "Agent")
    embs = [get_sentence_embedding(text) for text in texts]
    return torch.tensor(embs, dtype=torch.float32).unsqueeze(0)


def build_task_agent_runtime(
    task: Optional[Dict[str, Any]],
    default_agent_names: Sequence[str],
    default_node_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], torch.Tensor]:
    agent_names, node_kwargs = resolve_task_agent_config(task, default_agent_names, default_node_kwargs)
    agent_pool = build_agent_pool(agent_names, node_kwargs)
    agent_embeddings = build_agent_profile_embeddings_from_pool(agent_pool) # [1, 4, 384]
    for idx, emb in enumerate(agent_embeddings[0].detach().cpu().tolist()):
        if idx < len(agent_pool):
            agent_pool[idx]["agent_embedding"] = deepcopy(emb)
    return agent_names, node_kwargs, agent_pool, agent_embeddings

def get_local_idx(step: Union[int, TraceStep]) -> int:
    if isinstance(step, dict):
        if "local_idx" in step:
            return int(step["local_idx"])
        if "next_agent" in step:
            return int(step["next_agent"])
        raise KeyError("Trace step dict must contain 'local_idx' or 'next_agent'.")
    return int(step)


def trace_to_local_indices(trace: Optional[TraceLike]) -> List[int]:
    if trace is None:
        return []
    return [get_local_idx(step) for step in trace]


def build_agent_pool(
    agent_names: Sequence[str],
    node_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
    agent_embeddings: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    embeddings = _to_python_list(agent_embeddings)
    pool = []
    for idx, name in enumerate(agent_names):
        cfg = {}
        if node_kwargs is not None and idx < len(node_kwargs) and isinstance(node_kwargs[idx], dict):
            cfg = dict(node_kwargs[idx])
        role = cfg.get("role", name)
        desc = cfg.get("desc", cfg.get("description", role))
        emb = None
        if embeddings is not None and idx < len(embeddings):
            emb = deepcopy(embeddings[idx])
        pool.append(
            {
                "local_idx": idx,
                "agent_name": name,
                "agent_role": role,
                "agent_desc": desc,
                "agent_embedding": emb,
            }
        )
    return pool


def build_trace_step(
    local_idx: int,
    agent_pool: Sequence[Dict[str, Any]],
    step_idx: int,
    selection_score: Optional[float] = None,
    selection_prob: Optional[float] = None,
    allowed_by_graph: Optional[bool] = None,
    output_text: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> TraceStep:
    idx = int(local_idx)
    if idx < 0 or idx >= len(agent_pool):
        raise IndexError(f"local_idx {idx} out of range for agent_pool of size {len(agent_pool)}")
    agent = agent_pool[idx]
    return {
        "step_idx": int(step_idx),
        "local_idx": idx,
        "agent_name": agent.get("agent_name"),
        "agent_role": agent.get("agent_role"),
        "agent_desc": agent.get("agent_desc"),
        "agent_embedding": deepcopy(agent.get("agent_embedding")),
        "selection_score": None if selection_score is None else float(selection_score),
        "selection_prob": None if selection_prob is None else float(selection_prob),
        "allowed_by_graph": allowed_by_graph,
        "output_text": _to_python_list(output_text),
        "meta": {} if meta is None else deepcopy(meta),
    }


def build_trace_from_indices(
    indices: Sequence[int],
    agent_pool: Sequence[Dict[str, Any]],
) -> List[TraceStep]:
    return [build_trace_step(idx, agent_pool, step_idx=i) for i, idx in enumerate(indices)]


def trace_to_spatial_mask(trace: TraceLike, num_agents: int) -> List[List[int]]:
    adj = [[0 for _ in range(num_agents)] for _ in range(num_agents)]
    indices = trace_to_local_indices(trace)
    if not indices:
        return adj

    for i in range(len(indices) - 1):
        src = int(indices[i])
        dst = int(indices[i + 1])
        if 0 <= src < num_agents and 0 <= dst < num_agents and src != dst:
            adj[src][dst] = 1
    return adj


def trace_to_node_mask(trace: TraceLike, num_agents: int) -> List[int]:
    node_mask = [0 for _ in range(num_agents)]
    for idx in trace_to_local_indices(trace):
        if 0 <= int(idx) < num_agents:
            node_mask[int(idx)] = 1
    return node_mask


def trace_to_graph_spec(trace: TraceLike, num_agents: int) -> Dict[str, Any]:
    local_trace = trace_to_local_indices(trace)
    return {
        "spatial_mask": trace_to_spatial_mask(local_trace, num_agents),
        "temporal_mask": None,
        "node_mask": trace_to_node_mask(local_trace, num_agents),
        "selected_trace": local_trace,
    }


def trace_to_semantic_edges(trace: TraceLike) -> List[Dict[str, Any]]:
    semantic_trace = [step for step in trace if isinstance(step, dict)]
    edges = []
    for i in range(len(semantic_trace) - 1):
        src = semantic_trace[i]
        dst = semantic_trace[i + 1]
        edges.append(
            {
                "src_local_idx": int(src.get("local_idx", -1)),
                "dst_local_idx": int(dst.get("local_idx", -1)),
                "src_name": src.get("agent_name"),
                "dst_name": dst.get("agent_name"),
                "src_role": src.get("agent_role"),
                "dst_role": dst.get("agent_role"),
                "src_desc": src.get("agent_desc"),
                "dst_desc": dst.get("agent_desc"),
                "src_embedding": deepcopy(src.get("agent_embedding")),
                "dst_embedding": deepcopy(dst.get("agent_embedding")),
            }
        )
    return edges


def count_deadloops(trace: TraceLike, ngram: int = 2) -> int:
    indices = trace_to_local_indices(trace)
    if len(indices) < ngram:
        return 0
    grams = [tuple(indices[i:i + ngram]) for i in range(len(indices) - ngram + 1)]
    counter = Counter(grams)
    deadloops = 0
    for _, freq in counter.items():
        if freq > 1:
            deadloops += freq - 1
    return deadloops


def bootstrap_candidate_traces(num_agents: int) -> List[List[int]]:
    if num_agents <= 0:
        return []
    if num_agents == 1:
        return [[0]]

    agents = list(range(num_agents))
    last = num_agents - 1
    # traces = [
    #     agents,
    #     agents[:-1] + [max(0, last - 1), last],
    #     [0, 1, 0, last] if num_agents >= 2 else [0],
    #     [0, 1, 2, 1, last] if num_agents >= 4 else agents,
    #     [1, 0, 2, last] if num_agents >= 4 else agents,
    #     [1, 0, 2, 0, last] if num_agents >= 4 else agents,
    #     [0, 2, 1, last] if num_agents >= 4 else agents,
    # ]
    # 0=MathSolver, 1=ProblemDecomposer, 2=CalculationChecker, 3=ProgrammingExpert
    traces = [
        # [1, 0, 2],
        # [1, 0, 3, 2],
        # [1, 0, 2, 0],
        # [0, 2],
        # [1, 3, 0, 2],
        # [1, 0, 3, 2],
        # [0, 1, 0, 2],
        [1, 0, 2],
        [1, 0],
        [1, 0, 2, 0],
        [1, 0, 1, 0, 2],
        [1, 3, 0, 2],

    ]

    cleaned = []
    seen = set()
    for trace in traces:
        norm = tuple(int(max(0, min(num_agents - 1, x))) for x in trace)
        if norm not in seen:
            seen.add(norm)
            cleaned.append(list(norm))
    return cleaned


def aggregate_graphs(
    graphs: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float],
    num_agents: int,
) -> Tuple[List[float], List[List[float]]]:
    node_scores = [0.0 for _ in range(num_agents)]
    edge_scores = [[0.0 for _ in range(num_agents)] for _ in range(num_agents)]
    total_w = max(sum(float(w) for w in weights), 1e-8)

    for graph, weight in zip(graphs, weights):
        w = float(weight)
        for i in range(num_agents):
            row_active = False
            for j in range(num_agents):
                value = float(graph[i][j])
                edge_scores[i][j] += w * value
                if value > 0:
                    row_active = True
                    node_scores[i] += 0.5 * w
                    node_scores[j] += 0.5 * w
            if row_active:
                node_scores[i] += 0.5 * w

    node_scores = [min(1.0, max(0.0, s / total_w)) for s in node_scores]
    edge_scores = [
        [min(1.0, max(0.0, s / total_w)) for s in row]
        for row in edge_scores
    ]
    return node_scores, edge_scores
