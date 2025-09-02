import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .calibrator import CalibratorFactory
from .scaler import ScalerFactory
from .reshaper import ReshaperFactory
from .hook import HookManager

# --------------------------
# 量化器实现
# --------------------------

class QuantizerFactory:
    """校准器工厂"""
    @staticmethod
    def create(strategy: Dict):
        if strategy["quant_type"] == "fp8_e4m3":
            return BF16FP8Quantizer(strategy)
        elif strategy["quant_type"] == "int4":
            return FP8INT4Quantizer(strategy)
        else:
            print("未找到对应 quantizer")


class BaseQuantizer():
    def __init__(self, strategy):
        self.tensor_name = strategy["tensor_name"]
        self.strategy = strategy
        self.act_calibrator = self._init_calibrator("activation")
        self.weight_calibrator = self._init_calibrator("weight")
        self.quant_info = {}
        self.quant_weight = None
    
    def process(self):
        pass
    
    def to(self, device):
        self.weight_calibrator.to(device)
        self.act_calibrator.to(device)
    
    def register_params(self, layer, name, tensor):
        with torch.no_grad():
            if not hasattr(layer, name):
                layer.register_parameter(
                    name, 
                    torch.nn.Parameter(tensor, requires_grad=False)
                )
            else:
                setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))

        
    def _init_calibrator(self, param_type: str):
        """初始化校准器（需子类指定激活/权重使用的校准器类型）"""
        calibrator = CalibratorFactory.create(self.strategy.get(param_type).get("calibrator", "minmax"), \
                                              param_type, \
                                              self.strategy.get(param_type))
        return calibrator


class FP8INT4Quantizer(BaseQuantizer):
    def __init__(self, strategy):
        super().__init__(strategy)
    
    def calibrate(self, input_tensor, weight_tensor):
        # print(self.tensor_name)
        # print(self.strategy)
        # print(weight_tensor.dtype)
        # print(weight_tensor.device)
        self.act_calibrator.collect(input_tensor)
    
    def quantize(self, layer):
        # print(self.tensor_name)
        tensor_to_quant = self.dequant_fp8_tensor(layer)
        self.weight_calibrator.collect(tensor_to_quant)
        self.quant_info = self.compute_params()
        if self.strategy["weight"]["enable"]:
            self.quant_weight = self.weight_calibrator.quantize(tensor_to_quant, self.quant_info['weight'])
            self.quant_weight = self.pack_int4(self.quant_weight)
        self._register(layer)
    
    def compute_params(self):
        """直接返回校准器计算的参数（无额外调整）"""
        return {
            "activation": self.act_calibrator.compute_params(),
            "weight": self.weight_calibrator.compute_params()
        }
    
    
    def _register(self, layer):
        if self.strategy["weight"]["enable"]:
            self.register_params(layer, "weight", self.quant_weight)
            self.register_params(layer, "weight_scale_inv", self.quant_info["weight"]["scale"].squeeze(-1))
        if self.strategy["activation"]["enable"]:
            if self.quant_info["activation"]:
                self.register_params(layer, "input_scale", self.quant_info["activation"]["scale"].squeeze(-1))
    
    @staticmethod
    def dequant_fp8_tensor(layer):
        dequant_scaler = ScalerFactory.create(True, "bf16", "fp8_e4m3")
        dequant_reshaper = ReshaperFactory.create('block')
        tensor_to_dequant = layer.weight.data
        pre_quant_params = {"scale": layer.weight_scale_inv.data.unsqueeze(-1)}
        reshaped_tensor = dequant_reshaper.reshape(tensor_to_dequant)
        dequant_weight = dequant_scaler.dequantize(reshaped_tensor.float(), pre_quant_params)
        return dequant_reshaper.unreshape(dequant_weight)
    
    @staticmethod
    def pack_int4(tensor):
        pack_reshaper = ReshaperFactory.create('group')
        tensor = pack_reshaper.reshape(tensor, 2)
        high_bits = tensor[..., 1] & 0x0F
        low_bits = tensor[..., 0] & 0x0F
        packed_tensor = (high_bits << 4) | low_bits
        return packed_tensor



class BF16FP8Quantizer(BaseQuantizer):
    def __init__(self, strategy):
        super().__init__(strategy)
    
    def calibrate(self, input_tensor, weight_tensor):
        self.act_calibrator.collect(input_tensor)

    def quantize(self, layer):
        tensor_to_quant = layer.weight.data

        self.weight_calibrator.collect(tensor_to_quant)
        self.quant_info = self.compute_params()
        self.quant_weight = self.weight_calibrator.quantize(tensor_to_quant, self.quant_info['weight'])
        self._register(layer)
    
    def compute_params(self):
        """直接返回校准器计算的参数（无额外调整）"""
        return {
            "activation": self.act_calibrator.compute_params(),
            "weight": self.weight_calibrator.compute_params()
        }
    
    def _register(self, layer):
        if self.strategy["weight"]["enable"]:
            self.register_params(layer, "weight", self.quant_weight)
            self.register_params(layer, "weight_scale_inv", self.quant_info["weight"]["scale"].squeeze(-1))
        if self.strategy["activation"]["enable"]:
            self.register_params(layer, "input_scale", self.quant_info["activation"]["scale"].squeeze(-1))


    
    