import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .scaler import ScalerFactory
from .reshaper import ReshaperFactory

class BaseCalibrator:
    """校准器基类"""
    def __init__(self, sub_strategy):
        self.sub_strategy = sub_strategy
        self.granularity = sub_strategy["granularity"]
        self.stats = {}
        self.scaler = ScalerFactory.create(sub_strategy)
        self.reshaper = ReshaperFactory.create(sub_strategy.get("granularity", "channel"))
        self.enable = True
        self.initiated = False

    def collect(self, tensor: torch.Tensor, tensor_name: str) -> None:
        """收集张量统计信息"""
        raise NotImplementedError

    def compute_params(self, bit: int) -> Dict:
        """计算量化参数"""
        raise NotImplementedError

    def get_original_dtype(self) -> torch.dtype:
        """获取记录的原始数据类型"""
        pass


# --------------------------
# 校准器实现
# --------------------------
class CalibratorFactory:
    """校准器工厂"""
    @staticmethod
    def create(calibrator_type: str, sub_strategy: Dict) -> "BaseCalibrator":
        if calibrator_type == "minmax":
            return MinMaxCalibrator(sub_strategy)
        elif calibrator_type == "histogram":
            return HistogramCalibrator(sub_strategy)
        elif calibrator_type == "gptq":
            return GPTQCalibrator(sub_strategy)
        else:
            raise ValueError(f"不支持的校准器类型: {calibrator_type}")


class MinMaxCalibrator(BaseCalibrator):
    """MinMax校准器"""
    def __init__(self, sub_strategy):
        super().__init__(sub_strategy)
        self.max = None
        self.min = None
    
    def to(self, device):
        pass

    def initiate_calibrator(self, tensor):
        self.initiated = True
        if self.granularity == "dynamic":
            self.enable = False
            return
        
        # reshaped_tensor = self.reshaper.reshape(tensor)
        # other_dims = reshaped_tensor.shape[:-1]
        # other_dims = (*other_dims, 1)
        # self.min = torch.full(other_dims, float("inf"), dtype=tensor.dtype)
        # self.max = torch.full(other_dims, float("-inf"), dtype=tensor.dtype)

    def collect(self, tensor: torch.Tensor) -> None:  
        if not self.initiated:
            self.initiate_calibrator(tensor)
        
        if not self.enable:
            return

        if tensor.numel() == 0:
            self.enable = False
            return
        
        # 根据粒度收集统计信息
        reshaped_tensor = self.reshaper.reshape(tensor)
        current_min = reshaped_tensor.amin(dim=(-1), keepdim=True)
        current_max = reshaped_tensor.amax(dim=(-1), keepdim=True)

        # 更新统计信息

        # print(self.min)
        # print(self.min.shape) 
        # print(current_min)

        self.min = torch.min(self.min, current_min) if self.min is not None else current_min
        self.max = torch.max(self.max, current_max) if self.max is not None else current_max

    def compute_params(self) -> Dict[str, torch.Tensor]:
        """通过缩放器计算量化参数"""
        if self.enable:
        # 调用缩放器计算参数（自动适配对称/非对称和数据类型）
            return self.scaler.compute_params({"min": self.min, "max": self.max})
    
    def quantize(self, tensor):
        if not self.enable:
            return 
        # print(tensor.shape)
        reshaped_tensor = self.reshaper.reshape(tensor)
        # print(reshaped_tensor.shape)
        # print(self.scaler.params["scale"].shape)
        # print(self.scaler.params["offset"].shape)

        quant_weight = self.scaler.quantize(
            tensor=reshaped_tensor,
        )
        return self.reshaper.unreshape(quant_weight)

class HistogramCalibrator(BaseCalibrator):
    """直方图校准器（简化实现）"""
    def collect(self, tensor: torch.Tensor, tensor_name: str) -> None: 
        if tensor_name not in self.stats:
            self.stats[tensor_name] = []
        self.stats[tensor_name].append(tensor.detach().cpu())

    def compute_params(self) -> Dict[str, torch.Tensor]:
        """通过缩放器计算量化参数"""
        if not self.stats:
            # 未收集到统计信息时返回默认参数
            return {
                "scale": torch.tensor(1.0),
                "offset": torch.tensor(0, dtype=torch.int32)
            }
        
        # 调用缩放器计算参数（自动适配对称/非对称和数据类型）
        return self.scaler.compute_params({"min": self.min, "max": self.max})


class GPTQCalibrator(MinMaxCalibrator):
    """GPTQ专用校准器"""
    def __init__(self, granularity: str, group_size: int = 128):
        super().__init__(granularity)
        self.group_size = group_size

