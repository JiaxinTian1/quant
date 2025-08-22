import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .algo import AlgorithmFactory, BaseAlgorithm
from .calibrator import CalibratorFactory
from .hook import HookManager

# --------------------------
# 量化器实现
# --------------------------

class Quantizer():
    def __init__(self, strategy, tensor_name):
        self.tensor_name = tensor_name
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
    
    def calibrate(self, input_tensor, weight_tensor):
        print(self.tensor_name)
        # print(self.strategy)
        # print(weight_tensor.dtype)
        # print(weight_tensor.device)
        self.weight_calibrator.collect(weight_tensor)
        self.act_calibrator.collect(input_tensor)
        self.quant_info = self.compute_params()
    
    def quantize(self, layer):
        self.quant_weight = self.weight_calibrator.quantize(layer.weight.data)
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

        
    def _init_calibrator(self, param_type: str):
        """初始化校准器（需子类指定激活/权重使用的校准器类型）"""
        calibrator = CalibratorFactory.create(self.strategy.get(param_type).get("calibrator", "minmax"), \
                                              param_type, \
                                              self.strategy.get(param_type))
        return calibrator