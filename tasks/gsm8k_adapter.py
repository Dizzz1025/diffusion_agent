# tasks/gsm8k_adapter.py

from typing import Any, Dict, List

from GDesigner.tools.reader.readers import JSONLReader
from my_datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict

from tasks.base_adapter import BaseTaskAdapter


class GSM8KAdapter(BaseTaskAdapter):
    def load_tasks(self, dataset_json: str) -> List[Dict[str, Any]]:
        raw_dataset = JSONLReader.parse_file(dataset_json)
        dataset = gsm_data_process(raw_dataset)

        tasks = []
        for i, item in enumerate(dataset):
            tasks.append({
                "task_id": str(i),
                "task_type": "gsm8k",
                "task_text": item["task"],
                "ground_truth": item["answer"],
                "meta": {
                    "step": item.get("step", None)
                }
            })
        return tasks

    def build_input_dict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": task["task_text"]}

    def evaluate_answer(self, raw_answer: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        # 兼容 raw_answer 可能是 list / tuple
        answer_text = raw_answer[0] if isinstance(raw_answer, (list, tuple)) and len(raw_answer) > 0 else raw_answer
        predict_answer = gsm_get_predict(answer_text)

        gt = task["ground_truth"]
        correct = False
        if predict_answer is not None:
            try:
                correct = float(predict_answer) == float(gt)
            except Exception:
                correct = False

        return {
            "predict_answer": predict_answer,
            "correct": int(correct),
            "task_score": float(correct)
        }

    def get_task_text(self, task: Dict[str, Any]) -> str:
        return task["task_text"]