import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)

from model_adapter.deepseekv3.deepseekv3 import DSV3Model
from model_adapter.qwen3.qwen3 import Qwen3Model

class ModelAdapterFactory:
    """保存器工厂"""
    @staticmethod
    def create(model_type, model_path) -> "BaseSaver":
        if model_type == "qwen3":
            return Qwen3Model(model_path)
        elif model_type == "deepseekv3":
            return DSV3Model(model_path)