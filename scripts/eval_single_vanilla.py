from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 直接复用你仓库里的 LLM 封装与评测 adapter
from GDesigner.llm.gpt_chat import GPTChat
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens
from tasks.gsm8k_adapter import GSM8KAdapter


def snapshot_counters() -> Tuple[float, int, int]:
    return (
        Cost.instance().value,
        PromptTokens.instance().value,
        CompletionTokens.instance().value,
    )


def counter_delta(before: Tuple[float, int, int], after: Tuple[float, int, int]) -> Dict[str, float]:
    cost_before, pt_before, ct_before = before
    cost_after, pt_after, ct_after = after
    return {
        "cost": float(cost_after - cost_before),
        "prompt_tokens": int(pt_after - pt_before),
        "completion_tokens": int(ct_after - ct_before),
        "total_tokens": int((pt_after - pt_before) + (ct_after - ct_before)),
    }


def build_vanilla_messages(task_text: str) -> List[Dict[str, str]]:
    """
    严格单模型 vanilla：
    - 不走 graph
    - 不走多 agent
    - 不走 decision node
    - 不要求 step-by-step，避免变成 CoT
    """
    system_prompt = "You are a helpful assistant."
    user_prompt = (
        f"{task_text}\n\n"
        "Solve the problem and output only the final numerical answer."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def eval_one(
    llm: GPTChat,
    adapter: GSM8KAdapter,
    task: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:
    before = snapshot_counters()
    start_ts = time.time()

    messages = build_vanilla_messages(task["task_text"])
    raw_answer = await llm.agen(messages)

    eval_result = adapter.evaluate_answer(raw_answer, task)
    deltas = counter_delta(before, snapshot_counters())

    return {
        "idx": idx,
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "task_text": task["task_text"],
        "ground_truth": task["ground_truth"],
        "raw_answer": raw_answer,
        "predict_answer": eval_result["predict_answer"],
        "correct": int(eval_result["correct"]),
        "task_score": float(eval_result["task_score"]),
        "latency": time.time() - start_ts,
        **deltas,
    }


async def evaluate(args: argparse.Namespace) -> None:
    adapter = GSM8KAdapter()
    tasks = adapter.load_tasks(args.dataset_json)

    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]

    llm = GPTChat(model_name=args.llm_name)

    records: List[Dict[str, Any]] = []
    total_correct = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0.0

    print(f"[vanilla] loaded {len(tasks)} tasks from {args.dataset_json}")
    print(f"[vanilla] model = {args.llm_name}")
    tasks = tasks[:2]
    for i, task in enumerate(tasks):
        record = await eval_one(llm=llm, adapter=adapter, task=task, idx=i)
        records.append(record)

        total_correct += int(record["correct"])
        total_tokens += int(record["total_tokens"])
        total_prompt_tokens += int(record["prompt_tokens"])
        total_completion_tokens += int(record["completion_tokens"])
        total_latency += float(record["latency"])

        if (i + 1) % args.log_every == 0 or (i + 1) == len(tasks):
            acc = total_correct / max(1, len(records))
            print(
                f"[vanilla] {i+1}/{len(tasks)} | "
                f"acc={acc:.4f} | "
                f"avg_tokens={total_tokens / max(1, len(records)):.2f} | "
                f"avg_latency={total_latency / max(1, len(records)):.2f}s"
            )

    summary = {
        "method": "vanilla",
        "num_samples": len(records),
        "accuracy": total_correct / max(1, len(records)),
        "accuracy_percent": 100.0 * total_correct / max(1, len(records)),
        "avg_total_tokens": total_tokens / max(1, len(records)),
        "avg_prompt_tokens": total_prompt_tokens / max(1, len(records)),
        "avg_completion_tokens": total_completion_tokens / max(1, len(records)),
        "avg_latency": total_latency / max(1, len(records)),
        "model_name": args.llm_name,
        "dataset_json": args.dataset_json,
    }

    output = {
        "summary": summary,
        "records": records,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n===== VANILLA SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[vanilla] saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="my_datasets/gsm8k/gsm8k_test.jsonl",
        help="Path to GSM8K jsonl test set.",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="Meta-Llama-3.1-8B-Instruct",
        help="Model name passed to your OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/vanilla_gsm8k.json",
        help="Where to save evaluation results.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Use first N samples only. -1 means all.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print progress every N samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))