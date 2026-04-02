from typing import Any, Dict, List

from tasks.base_adapter import BaseTaskAdapter
from my_datasets.math_dataset import (
    load_math_records,
    math_check_correctness,
    math_data_process,
    math_get_predict,
)


class MathAdapter(BaseTaskAdapter):
    def load_tasks(self, dataset_json: str) -> List[Dict[str, Any]]:
        raw_dataset = load_math_records(dataset_json)
        dataset = math_data_process(raw_dataset)

        tasks = []
        for i, item in enumerate(dataset):
            tasks.append(
                {
                    "task_id": str(i),
                    "task_type": "math",
                    "task_text": item["task"],
                    "ground_truth": item["answer"],
                    "meta": {
                        "step": item.get("step"),
                        "level": item.get("level"),
                        "subject": item.get("subject"),
                    },
                }
            )
        return tasks

    def build_input_dict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": task["task_text"]}

    def evaluate_answer(self, raw_answer: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        answer_text = raw_answer[0] if isinstance(raw_answer, (list, tuple)) and len(raw_answer) > 0 else raw_answer
        predict_answer = math_get_predict(answer_text)
        gt = task["ground_truth"]
        correct = math_check_correctness(predict_answer, gt)

        return {
            "predict_answer": predict_answer,
            "correct": int(correct),
            "task_score": float(correct),
        }

    def get_task_text(self, task: Dict[str, Any]) -> str:
        return task["task_text"]
