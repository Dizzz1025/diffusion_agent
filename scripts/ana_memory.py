import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

memory  = 'memory_episode_100'
MEMORY_PATH = f"results/v3/{memory}.jsonl"
OUTPUT_JSON = f"results/v3/{memory}_analysis.json"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_value(x):
    """把 numpy / pandas 里的类型转成标准 JSON 可序列化类型。"""
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, tuple):
        return list(x)
    return x


def df_to_records(df):
    """DataFrame -> JSON 可保存的 records"""
    records = []
    for row in df.to_dict(orient="records"):
        new_row = {}
        for k, v in row.items():
            if isinstance(v, tuple):
                new_row[k] = list(v)
            elif isinstance(v, list):
                new_row[k] = [clean_value(i) for i in v]
            else:
                new_row[k] = clean_value(v)
        records.append(new_row)
    return records


rows = load_jsonl(MEMORY_PATH)
print(f"loaded memory items: {len(rows)}")

# -------- 基础表 --------
records = []
for i, x in enumerate(rows):
    trace = x.get("selected_trace", [])
    records.append({
        "idx": i,
        "task_text": x.get("task_text", ""),
        "trace": tuple(trace),
        "trace_str": "->".join(map(str, trace)),
        "reward": float(x.get("reward", 0.0)),
        "correct": int(x.get("correct", 0)),
        "token_cost": float(x.get("token_cost", 0.0)),
        "steps": int(x.get("steps", len(trace))),
        "num_agents": int(x.get("num_agents", 0)),
    })

df = pd.DataFrame(records)

print("\n===== 整体统计 =====")
overall_stats = {
    "memory_count": len(df),
    "unique_task": int(df["task_text"].nunique()),
    "unique_trace": int(df["trace"].nunique()),
    "correct_rate": float(df["correct"].mean()),
    "reward_mean": float(df["reward"].mean()),
    "reward_std": float(df["reward"].std()) if len(df) > 1 else None,
    "token_mean": float(df["token_cost"].mean()),
    "token_std": float(df["token_cost"].std()) if len(df) > 1 else None,
    "steps_mean": float(df["steps"].mean()),
    "steps_std": float(df["steps"].std()) if len(df) > 1 else None,
}
for k, v in overall_stats.items():
    print(k, ":", v)


# -------- 1) trace 频率分析 --------
trace_freq = (
    df.groupby(["trace", "trace_str"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

print("\n===== 最常见trace =====")
print(trace_freq.head(20))


# -------- 2) 按 trace pattern 分组统计 --------
trace_stats = (
    df.groupby(["trace", "trace_str"])
      .agg(
          count=("idx", "count"),
          acc=("correct", "mean"),
          avg_reward=("reward", "mean"),
          std_reward=("reward", "std"),
          avg_token=("token_cost", "mean"),
          avg_steps=("steps", "mean"),
      )
      .reset_index()
      .sort_values(["count", "avg_reward"], ascending=[False, False])
)

print("\n===== trace pattern统计 =====")
print(trace_stats.head(20))


# -------- 3) 按 steps 分组 --------
step_stats = (
    df.groupby("steps")
      .agg(
          count=("idx", "count"),
          acc=("correct", "mean"),
          avg_reward=("reward", "mean"),
          avg_token=("token_cost", "mean"),
      )
      .reset_index()
      .sort_values("steps")
)

print("\n===== steps统计 =====")
print(step_stats)


# # -------- 4) 同task保留2条memory的质量分析 --------
# task_groups = []
# for task_text, g in df.groupby("task_text"):
#     g = g.sort_values("reward", ascending=False).reset_index(drop=True)

#     traces = list(g["trace"])
#     rewards = list(g["reward"])
#     corrects = list(g["correct"])

#     duplicate_trace = len(set(traces)) < len(traces)
#     reward_gap = rewards[0] - rewards[1] if len(rewards) >= 2 else np.nan

#     task_groups.append({
#         "task_text": task_text,
#         "num_memories": len(g),
#         "duplicate_trace": duplicate_trace,
#         "top1_reward": rewards[0] if len(rewards) >= 1 else np.nan,
#         "top2_reward": rewards[1] if len(rewards) >= 2 else np.nan,
#         "reward_gap": reward_gap,
#         "top1_correct": corrects[0] if len(corrects) >= 1 else np.nan,
#         "top2_correct": corrects[1] if len(corrects) >= 2 else np.nan,
#         "trace1": "->".join(map(str, traces[0])) if len(traces) >= 1 else "",
#         "trace2": "->".join(map(str, traces[1])) if len(traces) >= 2 else "",
#     })

# task_df = pd.DataFrame(task_groups)

# print("\n===== 每个task的top2 memory分析 =====")
# print(task_df.head(20))

# duplicate_trace_ratio = float(task_df["duplicate_trace"].mean()) if len(task_df) > 0 else None
# reward_gap_mean = float(task_df["reward_gap"].mean()) if len(task_df) > 0 else None

# print("\n重复trace的task占比:", duplicate_trace_ratio)
# print("top1-top2 reward gap均值:", reward_gap_mean)

# combo_counter = Counter(zip(task_df["top1_correct"], task_df["top2_correct"]))
# print("top1/top2 correct组合统计:", combo_counter)


# -------- 5) 相关性分析 --------
print("\n===== 数值相关性 =====")
corr_cols = ["reward", "correct", "token_cost", "steps"]
corr_df = df[corr_cols].corr()
print(corr_df)


# -------- 6) selected_trace中的节点/边频率分析 --------
edge_counter = Counter()
node_counter = Counter()

for x in rows:
    g = x.get("support_graph", {})
    trace = g.get("selected_trace", x.get("selected_trace", []))

    if not isinstance(trace, (list, tuple)) or len(trace) == 0:
        continue

    node_counter.update(trace)
    edge_counter.update(zip(trace[:-1], trace[1:]))

print("\n===== 节点频率Top10 =====")
print(node_counter.most_common(10))

print("\n===== 边频率Top20 =====")
print(edge_counter.most_common(20))


# -------- 保存 JSON --------
analysis = {
    "memory_path": MEMORY_PATH,
    "overall_stats": overall_stats,

    "trace_freq_top20": df_to_records(trace_freq.head(20)),
    "trace_stats_top20": df_to_records(trace_stats.head(20)),
    "step_stats": df_to_records(step_stats),

    # "task_top2_head20": df_to_records(task_df.head(20)),
    # "task_summary": {
    #     "duplicate_trace_ratio": duplicate_trace_ratio,
    #     "reward_gap_mean": reward_gap_mean,
    #     "top1_top2_correct_combo": {
    #         f"{clean_value(k1)}_{clean_value(k2)}": int(v)
    #         for (k1, k2), v in combo_counter.items()
    #     }
    # },

    "correlation": {
        row: {col: clean_value(corr_df.loc[row, col]) for col in corr_df.columns}
        for row in corr_df.index
    },

    "node_freq_top10": [
        {"node": clean_value(node), "count": int(cnt)}
        for node, cnt in node_counter.most_common(10)
    ],
    "edge_freq_top20": [
        {"edge": f"{u}->{v}", "u": clean_value(u), "v": clean_value(v), "count": int(cnt)}
        for (u, v), cnt in edge_counter.most_common(20)
    ],
}

Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"\n分析结果已保存到: {OUTPUT_JSON}")