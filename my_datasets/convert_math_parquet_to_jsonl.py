import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_record(row: dict) -> dict:
    """Convert one parquet row to the JSONL schema used by the v3 training pipeline."""
    problem = str(row.get("problem", "")).strip()
    solution = str(row.get("solution", "")).strip()
    answer = str(row.get("answer", "")).strip()
    subject = str(row.get("subject", "")).strip()
    unique_id = str(row.get("unique_id", "")).strip()

    level = row.get("level", None)
    try:
        level = int(level) if level is not None and str(level) != "" else None
    except Exception:
        level = None

    # Keep both the original MATH fields and a generic task_text field for convenience.
    return {
        "id": unique_id,
        "task_text": problem,
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "subject": subject,
        "level": level,
        "unique_id": unique_id,
        "dataset": "MATH",
    }



def convert_one_file(input_path: Path, output_path: Path) -> int:
    df = pd.read_parquet(input_path)
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            record = normalize_record(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count



def convert_dir(input_dir: Path, output_path: Path) -> int:
    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {input_dir}")

    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for row in df.to_dict(orient="records"):
                record = normalize_record(row)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    return total



def main():
    input_path = Path("/root/autodl-tmp/diffusion_agent/my_datasets/MATH/train-00000-of-00001.parquet")
    output_path = Path("/root/autodl-tmp/diffusion_agent/my_datasets/MATH/train.jsonl")

    if input_path.is_file():
        n = convert_one_file(input_path, output_path)
    elif input_path.is_dir():
        n = convert_dir(input_path, output_path)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Done. Wrote {n} records to {output_path}")


if __name__ == "__main__":
    main()
