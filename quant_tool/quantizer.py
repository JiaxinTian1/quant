import torch
import torch.nn as nn
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

from .calibrator import CalibratorFactory
from .scaler import ScalerFactory
from .reshaper import ReshaperFactory
from .algorithm import AlgorithmFactory


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
        self.algorithm = AlgorithmFactory.create(strategy)
        self.quant_info = {}
        self.quant_weight = None

        self.quant_status = "post_calib"
        self.need_preprocess = False
        self.pre_calibrated = False
        self.post_calibrated = False
    
    def process(self):
        pass
    
    def to(self, device):
        pass

    def pre_compute(self):
        if self.need_preprocess and self.pre_calibrated:
            self.algorithm.pre_compute()

    def post_compute(self):
        if self.post_calibrated:
            self.algorithm.post_compute()
    
    def register_params(self, layer, name, tensor):
        with torch.no_grad():
            if not hasattr(layer, name):
                layer.register_parameter(
                    name, 
                    torch.nn.Parameter(tensor, requires_grad=False)
                )
            else:
                setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))


class FP8INT4Quantizer(BaseQuantizer):
    def __init__(self, strategy):
        super().__init__(strategy)
    
    def calibrate(self, input_tensor, weight_module):
        weight_tensor = self.dequant_fp8_tensor(weight_module)
        if self.quant_status == "post_calib":
            if self.need_preprocess and not self.pre_calibrated:
                return
            self.algorithm.post_calib(input_tensor, weight_tensor)
            self.post_calibrated = True

        elif self.quant_status == "pre_calib":
            self.algorithm.pre_calib(input_tensor, weight_tensor)
            self.pre_calibrated = True
    
    def quantize(self, layer):
        tensor_to_quant = self.dequant_fp8_tensor(layer)
        if not self.post_calibrated:
            tensor_to_quant = self.dequant_fp8_tensor(layer)
            self.algorithm.weight_calibrator.collect(tensor_to_quant)
            self.post_calibrated = True
        self.post_compute()
        self.quant_info = self.algorithm.post_compute_info
        if self.strategy["weight"]["enable"]:
            if self.post_calibrated:
                self.quant_weight = self.algorithm.weight_calibrator.quantize(tensor_to_quant, self.quant_info['weight'])
                self.quant_weight = self.pack_int4(self.quant_weight)
        self._register(layer)
    
    # def compute_params(self):
    #     current_granularity = self.strategy["activation"]["granularity"]
    #     if current_granularity not in ["tensor"]:
    #         raise NotImplementedError(f"不支持的粒度级别: {current_granularity}。")
        
    #     act_params = self.act_calibrator.compute_params()
    #     weight_params = self.weight_calibrator.compute_params()
    #     if act_params is None or weight_params is None:
    #         return {
    #             "activation": act_params,
    #             "weight": weight_params
    #         }
    #     act_amax = torch.max(torch.abs(self.act_calibrator.max), torch.abs(self.act_calibrator.min))
    #     weight_amax = torch.max(torch.abs(self.weight_calibrator.max), torch.abs(self.weight_calibrator.min))
        
    #     eps = 1e-8
    #     alpha = 0.5

    #     # 使用weight_amax的平均值替代原来的s_group平均值
    #     weight_amax_max = torch.max(weight_amax)
    #     # 或者使用weight_amax的最大值
    #     # weight_amax_max = torch.max(weight_amax)
    #     s_group = (act_amax ** alpha) / ((weight_amax_max + eps) ** (1 - alpha))

    #     adjusted_act = act_params.copy()
    #     adjusted_act["scale"] /= (s_group  + eps)

    #     adjusted_weight = weight_params.copy()
    #     adjusted_weight["scale"] *= (s_group  + eps)
    #     # breakpoint()
    #     return {"activation": adjusted_act, "weight": adjusted_weight}
        

    def _register(self, layer):
        if self.strategy["weight"]["enable"]:
            self.register_params(layer, "weight", self.quant_weight)
            if hasattr(layer, 'weight_scale_inv'):
                self.register_params(layer, 'weight_scale_inv', self.quant_info["weight"]["scale"].squeeze(-1))
            else:
                self.register_params(layer, 'scale', self.quant_info["weight"]["scale"].squeeze(-1))
        if self.strategy["activation"]["enable"]:
            if self.quant_info["activation"]:
                self.register_params(layer, "input_scale", self.quant_info["activation"]["scale"].squeeze(-1))
    
    def dequant_fp8_tensor(self, layer):
        weight_original_dtype = self.strategy["weight"]["original_dtype"]
        dequant_scaler = ScalerFactory.create(True, weight_original_dtype, "fp8_e4m3")
        dequant_reshaper = ReshaperFactory.create('block')
        tensor_to_dequant = layer.weight.data
        if hasattr(layer, 'weight_scale_inv'):
            pre_quant_params = {"scale": layer.weight_scale_inv.data.unsqueeze(-1)}
        else:
            pre_quant_params = {"scale": layer.scale.data.unsqueeze(-1)}
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

    
    def calibrate(self, input_tensor, weight_module):
        weight_tensor = weight_module.weight.data
        if self.quant_status == "post_calib":
            if self.need_preprocess and not self.pre_calibrated:
                return
            self.algorithm.post_calib(input_tensor, weight_tensor)
            self.post_calibrated = True

        elif self.quant_status == "pre_calib":
            self.algorithm.pre_calib(input_tensor, weight_tensor)
            self.pre_calibrated = True

    def quantize(self, layer):
        tensor_to_quant = layer.weight.data
        if not self.post_calibrated:
            self.algorithm.weight_calibrator.collect(tensor_to_quant)
            self.post_calibrated = True
        self.post_compute()
        self.quant_info = self.algorithm.post_compute_info
        if self.strategy["weight"]["enable"]:
            self.quant_weight = self.algorithm.weight_calibrator.quantize(tensor_to_quant, self.quant_info['weight'])
        self._register(layer)

    def preprocess(self, input_tensor, weight_module):
        processed_input = input_tensor
        processed_weight = weight_module.weight.data
        if self.need_preprocess:
            processed_input, processed_weight = self.algorithm.preprocess(
                processed_input, processed_weight)
        return processed_input, processed_weight
    
    def _register(self, layer):
        if self.strategy["weight"]["enable"]:
            self.register_params(layer, "weight", self.quant_weight)
            self.register_params(layer, "weight_scale_inv", self.quant_info["weight"]["scale"].squeeze(-1))
        if self.strategy["activation"]["enable"]:
            self.register_params(layer, "input_scale", self.quant_info["activation"]["scale"].squeeze(-1))

    