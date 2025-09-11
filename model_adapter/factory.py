from .deepseekv3.deepseekv3 import DSV3Model
from .qwen3.qwen3 import Qwen3Model


class ModelAdapterFactory:
    """模型适配器工厂，创建不同类型的模型适配器"""
    @staticmethod
    def create(model_type: str,  model_path: str, save_path: str, dist_ctx):
        """
        创建模型适配器实例
        
        Args:
            model_type: 模型类型（如"qwen3"、"deepseekv3"）
            model_path: 模型文件路径
            dist_ctx: 分布式上下文对象
            
        Returns:
            模型适配器实例
        """
        if model_type in ["qwen3_moe", "qwen3"]:
            return Qwen3Model(model_path, save_path, dist_ctx)
        elif model_type == "deepseek_v3":
            return DSV3Model(model_path, save_path, dist_ctx)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")