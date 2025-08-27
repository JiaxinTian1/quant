import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .scaler import ScalerFactory
from .reshaper import ReshaperFactory

class BaseCalibrator:
    """校准器基类"""
    def __init__(self, tensor_type, sub_strategy):
        self.sub_strategy = sub_strategy
        self.tensor_type = tensor_type
        self.stats = {}
        self.enable = True
        self.initiated = False
        self.collected = False
        self.reshaper = None
        self.scaler = None

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
    def create(calibrator_type: str, tensor_type, sub_strategy: Dict) -> "BaseCalibrator":
        if calibrator_type == "minmax":
            return MinMaxCalibrator(tensor_type, sub_strategy)
        elif calibrator_type == "histogram":
            return HistogramCalibrator(tensor_type, sub_strategy)
        elif calibrator_type == "gptq":
            return GPTQCalibrator(tensor_type, sub_strategy)
        else:
            raise ValueError(f"不支持的校准器类型: {calibrator_type}")


class MinMaxCalibrator(BaseCalibrator):
    """MinMax校准器"""
    def __init__(self, tensor_type, sub_strategy):
        super().__init__(tensor_type, sub_strategy)
        self.max = None
        self.min = None
        self.initiate_calibrator()
    
    def to(self, device):
        pass

    def initiate_calibrator(self):
        self.initiated = True
        if not self.sub_strategy.get("enable", False):
            self.enable = False
        if self.enable:
            self.scaler = ScalerFactory.create(
                self.sub_strategy.get("is_sym", True),
                self.sub_strategy.get("original_dtype"),
                self.sub_strategy.get("target_dtype"),
                )
            self.reshaper = ReshaperFactory.create(self.sub_strategy.get("granularity", "channel"))

    def collect(self, tensor: torch.Tensor) -> None:  
        if not self.enable:
            return

        # 根据粒度收集统计信息
        reshaped_tensor = self.reshaper.reshape(tensor)
        # breakpoint()
        current_min, current_max = self._get_tensor_feature(reshaped_tensor)

        # 更新统计信息

        # print(self.min)
        # print(self.min.shape) 
        # print(current_min)

        self.min = torch.min(self.min, current_min) if self.min is not None else current_min
        self.max = torch.max(self.max, current_max) if self.max is not None else current_max
        self.collected = True
        
    def compute_params(self) -> Dict[str, torch.Tensor]:
        """通过缩放器计算量化参数"""
        if self.enable:
            # 调用缩放器计算参数（自动适配对称/非对称和数据类型）
            if not self.collected:
                return
            return self.scaler.compute_params({"min": self.min, "max": self.max})
    
    def quantize(self, tensor, params):
        if not self.enable:
            return 
        # print(tensor.shape)
        # print(self.reshaper)
        reshaped_tensor = self.reshaper.reshape(tensor)
        # print(reshaped_tensor.shape)
        # print(self.scaler.params["scale"].shape)
        # print(self.scaler.params["offset"].shape)
        
        quant_weight = self.scaler.quantize(reshaped_tensor.float(), params)
        return self.reshaper.unreshape(quant_weight)
    
    def dequantize(self, tensor, params):
        if not self.enable:
            return 
        # print(tensor.shape)
        # print(self.reshaper)
        reshaped_tensor = self.reshaper.reshape(tensor)
        # print(reshaped_tensor.shape)
        # print(self.scaler.params["scale"].shape)
        # print(self.scaler.params["offset"].shape)
        
        dequant_weight = self.scaler.dequantize(reshaped_tensor.float(), params)
        return self.reshaper.unreshape(dequant_weight)
    
    def _get_tensor_feature(self, tensor):
        if self.tensor_type == 'weight':
            current_min = tensor.float().amin(dim=(-1), keepdim=True)
            current_max = tensor.float().amax(dim=(-1), keepdim=True)
        elif self.tensor_type == 'activation':
            # 获取所有维度的索引
            all_dims = tuple(range(tensor.dim()))
            # 排除-2维度（倒数第二维）
            dims_to_reduce = tuple(d for d in all_dims if d not in [tensor.dim() - 2, tensor.dim() - 1])
            current_min = tensor.amin(dim=dims_to_reduce).amin(dim=(-1), keepdim=True)
            current_max = tensor.amax(dim=dims_to_reduce).amax(dim=(-1), keepdim=True)
        # breakpoint()
        return (current_min, current_max)

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

