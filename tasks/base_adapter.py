# tasks/base_adapter.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTaskAdapter(ABC):
    @abstractmethod
    def load_tasks(self, dataset_json: str) -> List[Dict[str, Any]]:
        """
        读取原始数据集，并转成统一 task schema
        """
        raise NotImplementedError

    @abstractmethod
    def build_input_dict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        给 GDesigner Graph.arun() 构造输入
        例如 {"task": "..."}
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_answer(self, raw_answer: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析 raw_answer，计算 predict_answer / correct / task_score
        """
        raise NotImplementedError

    @abstractmethod
    def get_task_text(self, task: Dict[str, Any]) -> str:
        """
        返回用于 embedding / memory 检索的文本
        """
        raise NotImplementedError