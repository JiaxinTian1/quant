import torch
import torch.nn as nn
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class BaseScaler(ABC):
    """缩放器基类，定义量化参数计算接口"""
        
    @abstractmethod
    def compute_params(self, stats: Dict) -> Dict:
        pass
    
    @abstractmethod
    def quantize(self, tensor: torch.Tensor, params: Dict) -> torch.Tensor:
        pass
    
    @abstractmethod
    def dequantize(self, tensor: torch.Tensor, params: Dict) -> torch.Tensor:
        pass


# --------------------------
# 缩放器实现
# --------------------------
class ScalerFactory:
    """缩放器工厂"""
    @staticmethod
    def create(is_sym, original_dtype, target_dtype) -> "BaseScaler":
        if is_sym:
            return SymmetricScaler(original_dtype, target_dtype)
        else:
            return AsymmetricScaler(original_dtype, target_dtype)


class SymmetricScaler(BaseScaler):
    """对称缩放器：适用于所有对称量化场景（支持int4/int8/int16等）"""
    TYPE_RANGE = {
        # 有符号整数类型
        "int4": (-8, 7),
        "int8": (-128, 127),
        "fp8_e4m3": (-448, 448), 
        "bf16": (-128, 127),
    }

    def __init__(self, original_dtype, target_dtype):
        self.original_dtype = original_dtype
        self.target_dtype = target_dtype
        self.d_min, self.d_max = self.TYPE_RANGE[original_dtype]
        self.q_min, self.q_max = self.TYPE_RANGE[target_dtype]

    def compute_params(self, stats: Dict) -> Dict:
        min_val = stats["min"]
        max_val = stats["max"]

        # 对称量化取绝对值最大者作为动态范围
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        
        # 计算scale：原始动态范围 / 量化范围的一半（因对称）
        # 例：int8的量化范围是[-128, 127]，有效范围为127 - (-128) = 255，但对称场景用127作为分母
        quant_range_half = self.q_max  # 对称量化的有效范围上限（如int8为127）
        scale = max_abs / quant_range_half
        
        # 避免除零
        scale = torch.clamp(scale, min=1e-8)
        return {"scale": scale}
    
    def quantize(self, tensor: torch.Tensor, params):
        scale = params['scale']
        tensor_quant = (tensor / scale)
        if self.target_dtype in ["int4", "int8"]:
            tensor_quant = torch.round(tensor_quant)
        return torch.clamp(
            tensor_quant,
            min=self.q_min,
            max=self.q_max
        ).to(parse_dtype(self.target_dtype))
    
    def dequantize(self, tensor: torch.Tensor, params):
        scale = params['scale']
        tensor_quant = (tensor) * scale
        return torch.clamp(
            tensor_quant,
            min=self.d_min,
            max=self.d_max
        ).to(parse_dtype(self.original_dtype))
    


class AsymmetricScaler(BaseScaler):
    """非对称缩放器：适用于所有非对称量化场景（支持int4/int8/int16等）"""
    TYPE_RANGE = {
        # 无符号整数类型（较少用于量化，但可扩展）
        "int4": (0, 15),
        "int8": (0, 255),
    }

    def __init__(self, original_dtype, target_dtype):
        self.original_dtype = original_dtype
        self.target_dtype = target_dtype
        self.d_min, self.d_max = self.TYPE_RANGE[original_dtype]
        self.q_min, self.q_max = self.TYPE_RANGE[target_dtype]
    
    def compute_params(self, stats: Dict) -> Dict:
        min_val = stats["min"]
        max_val = stats["max"]
        
        # 非对称量化动态范围为原始数据的[min, max]
        data_range = max_val - min_val
        # 量化范围为 [q_min, q_max] 的差值
        quant_range = self.q_max - self.q_min
        
        # 计算scale
        scale = data_range / quant_range
        scale = torch.clamp(scale, min=1e-8)
        
        # 计算offset（偏移量）：确保原始min映射到q_min
        offset = self.q_min - torch.round(min_val / scale)
        # 裁剪到量化范围内（避免超出q_min/q_max）
        offset = torch.clamp(offset, self.q_min, self.q_max).to(torch.int32)
        return {"scale": scale, "offset": offset}


# --------------------------
# 工具函数
# --------------------------
def parse_dtype(dtype_str: str) -> Tuple[torch.dtype, int]:
    """解析量化类型字符串为torch.dtype和位数"""
    dtype_map = {
        "int4": torch.int8,  # 用uint8存储int4
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "fp8_e4m3": torch.float8_e4m3fn,
        "bf16": torch.bfloat16
    }
    dtype_str = dtype_str.lower()
    if dtype_str not in dtype_map:
        raise ValueError(f"不支持的量化类型: {dtype_str}")
    return dtype_map[dtype_str]