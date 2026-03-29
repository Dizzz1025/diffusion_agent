from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from memory.trajectory_memory_bank import TrajectoryMemoryBank
from policy.graph_generator import GraphGenerator, graph_generator_loss
from utils.v3_trace_utils import build_task_agent_runtime


def to_tensor_summary(summary: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "node_prior": torch.tensor(summary["node_prior"], dtype=torch.float32, device=device).unsqueeze(0),
        "edge_prior": torch.tensor(summary["edge_prior"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def build_runtime_from_item(item: Dict[str, Any]):
    """
    根据 memory item 里的 agent_pool 重建当前 task 的 agent runtime。
    这样和 train_v3.py 的逻辑保持一致。
    """
    item_agent_pool = item.get("agent_pool", [])
    fallback_names = [x.get("agent_name", f"Agent{i}") for i, x in enumerate(item_agent_pool)]
    fallback_node_kwargs = [
        {
            "role": x.get("agent_role", x.get("agent_name", f"Agent{i}")),
            "desc": x.get("agent_desc", x.get("agent_role", x.get("agent_name", f"Agent{i}"))),
        }
        for i, x in enumerate(item_agent_pool)
    ]

    agent_names, node_kwargs, agent_pool, agent_profile_embeddings = build_task_agent_runtime(
        {"agent_pool": item_agent_pool},
        fallback_names,
        fallback_node_kwargs,
    )
    return agent_names, node_kwargs, agent_pool, agent_profile_embeddings


def flatten_offdiag(mat: torch.Tensor) -> torch.Tensor:
    """
    mat: [N, N] or [1, N, N]
    取非对角元素，避免 self-loop 的 1e4 mask 干扰评价。
    """
    if mat.dim() == 3:
        mat = mat[0]
    n = mat.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat[~eye]


def binary_prf(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    pred = pred.bool()
    target = target.bool()

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def evaluate_one_item(
    idx: int,
    item: Dict[str, Any],
    memory_bank: TrajectoryMemoryBank,
    model: GraphGenerator,
    device: torch.device,
    top_k: int,
    node_threshold: float,
    edge_threshold: float,
    leave_one_out: bool = True,
) -> Dict[str, Any]:
    _, _, agent_pool, agent_profile_embeddings = build_runtime_from_item(item)
    agent_profile_embeddings = agent_profile_embeddings.to(device)

    exclude_index = idx if leave_one_out else None

    summary = memory_bank.summarize(
        task_embedding=item["task_embedding"],
        current_agent_pool=agent_pool,
        top_k=top_k,
        exclude_index=exclude_index,
    )
    labels = memory_bank.build_soft_labels(
        task_embedding=item["task_embedding"],
        current_agent_pool=agent_pool,
        top_k=top_k,
        fallback_item=item,
        exclude_index=exclude_index,
    )

    summary_tensors = to_tensor_summary(summary, device)
    task_tensor = torch.tensor(item["task_embedding"], dtype=torch.float32, device=device).unsqueeze(0)
    node_targets = torch.tensor(labels["node_targets"], dtype=torch.float32, device=device).unsqueeze(0)
    edge_targets = torch.tensor(labels["edge_targets"], dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(task_tensor, agent_profile_embeddings, memory_summary=summary_tensors)
        pred = model.predict_graph(
            task_tensor,
            agent_profile_embeddings,
            memory_summary=summary_tensors,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        loss = graph_generator_loss(outputs, node_targets, edge_targets).item()

    # 概率回归误差
    node_mae = torch.abs(outputs["node_probs"] - node_targets).mean().item()
    edge_mae = torch.abs(flatten_offdiag(outputs["edge_probs"]) - flatten_offdiag(edge_targets)).mean().item()

    # 二值化评价（soft label > 0.5 视为正样本）
    node_target_bin = (node_targets >= 0.5)[0]
    edge_target_bin = flatten_offdiag(edge_targets >= 0.5)

    node_pred_bin = pred["node_mask"][0] > 0
    edge_pred_bin = flatten_offdiag(pred["edge_mask"] > 0)

    node_p, node_r, node_f1 = binary_prf(node_pred_bin, node_target_bin)
    edge_p, edge_r, edge_f1 = binary_prf(edge_pred_bin, edge_target_bin)

    return {
        "idx": idx,
        "task_text": item.get("task_text", ""),
        "loss": loss,
        "node_mae": node_mae,
        "edge_mae": edge_mae,
        "node_precision": node_p,
        "node_recall": node_r,
        "node_f1": node_f1,
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "edge_f1": edge_f1,
        "node_probs": outputs["node_probs"][0].detach().cpu().tolist(),
        "edge_probs": outputs["edge_probs"][0].detach().cpu().tolist(),
        "node_targets": labels["node_targets"],
        "edge_targets": labels["edge_targets"],
        "retrieved_count": len(summary.get("retrieved", [])),
        "agent_pool": agent_pool,
    }


def print_case(result: Dict[str, Any], top_edges: int = 8):
    print("=" * 100)
    print(f"[case #{result['idx']}]")
    print("task_text:", result["task_text"][:200].replace("\n", " "))
    print(
        f"loss={result['loss']:.4f} | "
        f"node_mae={result['node_mae']:.4f} | edge_mae={result['edge_mae']:.4f} | "
        f"node_f1={result['node_f1']:.4f} | edge_f1={result['edge_f1']:.4f} | "
        f"retrieved={result['retrieved_count']}"
    )

    roles = [x.get("agent_role", x.get("agent_name", f"Agent{i}")) for i, x in enumerate(result["agent_pool"])]

    print("\n[node probs]")
    for i, (role, p, t) in enumerate(zip(roles, result["node_probs"], result["node_targets"])):
        print(f"  {i:>2d} | {role:<24} pred={p:.4f}  target={t:.4f}")

    edge_list = []
    edge_probs = result["edge_probs"]
    edge_targets = result["edge_targets"]
    n = len(edge_probs)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_list.append((i, j, edge_probs[i][j], edge_targets[i][j]))

    edge_list.sort(key=lambda x: x[2], reverse=True)

    print("\n[top predicted edges]")
    for i, j, p, t in edge_list[:top_edges]:
        print(
            f"  {i}->{j} | "
            f"{roles[i]} -> {roles[j]} | "
            f"pred={p:.4f} target={t:.4f}"
        )


def average_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}

    keys = [
        "loss",
        "node_mae",
        "edge_mae",
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_precision",
        "edge_recall",
        "edge_f1",
    ]
    out = {}
    for k in keys:
        out[k] = sum(x[k] for x in results) / len(results)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-jsonl", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="graph_generator.pt")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--node-threshold", type=float, default=0.45)
    parser.add_argument("--edge-threshold", type=float, default=0.45)
    parser.add_argument("--max-cases", type=int, default=10, help="最多评估多少条 memory item；-1 表示全部")
    parser.add_argument("--show-cases", type=int, default=3, help="打印多少条样例")
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    memory_bank = TrajectoryMemoryBank(max_size=100000)
    memory_bank.load_jsonl(args.memory_jsonl)

    if len(memory_bank) == 0:
        raise ValueError(f"memory bank is empty: {args.memory_jsonl}")

    first_item = memory_bank.items[0]
    _, _, first_agent_pool, first_agent_embs = build_runtime_from_item(first_item)

    task_dim = len(first_item["task_embedding"])
    agent_dim = first_agent_embs.size(-1)
    num_agents = len(first_agent_pool)

    model = GraphGenerator(
        task_dim=task_dim,
        agent_dim=agent_dim,
        hidden_dim=args.hidden_dim,
        num_agents=num_agents,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    items = memory_bank.items
    if args.max_cases > 0:
        items = items[:args.max_cases]

    results = []
    for idx, item in enumerate(items):
        res = evaluate_one_item(
            idx=idx,
            item=item,
            memory_bank=memory_bank,
            model=model,
            device=device,
            top_k=args.top_k,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            leave_one_out=args.leave_one_out,
        )
        results.append(res)

    avg = average_metrics(results)

    print("\n" + "#" * 100)
    print("[overall]")
    for k, v in avg.items():
        print(f"{k}: {v:.6f}")

    print("\n" + "#" * 100)
    print("[sample predictions]")
    for res in results[:args.show_cases]:
        print_case(res)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "average": avg,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nsaved to: {save_path}")


if __name__ == "__main__":
    main()