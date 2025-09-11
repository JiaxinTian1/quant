import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class BaseMethod(ABC):
    def __init__(self, strategy):
        self.strategy = strategy
    
    @abstractmethod
    def preprocess_quant(self, layer):
        pass
    
    @abstractmethod
    def postprocess_quant(self, layer, quant_results, quant_info):
        pass 
    
    @abstractmethod
    def register(self, layer, quant_results, quant_info):
        pass

class QuantMethodFactory():
    """Algorithm工厂"""
    @staticmethod
    def create(method) -> "BaseMethod":
        if method == 'minmax':
            return BF16FP8Method(strategy)
        elif method == 'smoothquant':
            return FP8INT4Method(strategy)


class FP8INT4Method(QuantMethod):
    def preprocess_quant(self, layer):
        """FP8转FP32（解量化）"""
        # 复用原有dequant_fp8_tensor逻辑
        weight_original_dtype = layer.weight.dtype
        dequant_scaler = ScalerFactory.create(True, weight_original_dtype, "fp8_e4m3")
        dequant_reshaper = ReshaperFactory.create('block')
        tensor_to_dequant = layer.weight.data
        
        # 获取预量化参数
        if hasattr(layer, 'weight_scale_inv'):
            pre_quant_params = {"scale": layer.weight_scale_inv.data.unsqueeze(-1)}
        else:
            pre_quant_params = {"scale": layer.scale.data.unsqueeze(-1)}
        
        # 执行解量化
        reshaped_tensor = dequant_reshaper.reshape(tensor_to_dequant)
        dequant_weight = dequant_scaler.dequantize(reshaped_tensor.float(), pre_quant_params)
        return dequant_reshaper.unreshape(dequant_weight)
    
    def postprocess_quant(self, layer, quant_weight, quant_info):
        """INT4打包 + 参数注册"""
        # 1. 打包INT4权重
        pack_reshaper = ReshaperFactory.create('group')
        tensor = pack_reshaper.reshape(quant_weight, 2)
        high_bits = tensor[..., 1] & 0x0F
        low_bits = tensor[..., 0] & 0x0F
        packed_tensor = (high_bits << 4) | low_bits
        
        # 2. 注册参数到layer
        if hasattr(layer, 'weight_scale_inv'):
            layer.register_buffer('weight_scale_inv', quant_info["weight"]["scale"].squeeze(-1))
        else:
            layer.register_buffer('scale', quant_info["weight"]["scale"].squeeze(-1))
        
        if quant_info.get("activation"):
            layer.register_buffer('input_scale', quant_info["activation"]["scale"].squeeze(-1))
            
        layer.weight.data = packed_tensor


class BF16FP8Method(QuantMethod):
    def preprocess_quant(self, layer):
        """BF16无需解量化，直接返回原始权重"""
        return layer.weight.data
    
    def postprocess_quant(self, layer, quant_weight, quant_info):
        """无需打包，直接注册参数"""
        # 注册权重和参数
        layer.register_buffer('weight', quant_weight)
        layer.register_buffer('weight_scale_inv', quant_info["weight"]["scale"].squeeze(-1))
        
        if quant_info.get("activation"):
            layer.register_buffer('input_scale', quant_info["activation"]["scale"].squeeze(-1))
            
