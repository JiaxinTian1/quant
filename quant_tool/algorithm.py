import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .calibrator import CalibratorFactory


class BaseAlgorithm(ABC):
    """量化算法策略接口，所有具体算法（SmoothQuant/AWQ）需实现此类"""
    def __init__(self, strategy):
        self.tensor_name = strategy["tensor_name"]
        self.strategy = strategy
        self.act_calibrator = self._init_calibrator("activation")
        self.weight_calibrator = self._init_calibrator("weight")
        self.post_compute_info = {}
    
    @abstractmethod
    def pre_calib(self, input_tensor, weight_tensor):
        pass
    
    @abstractmethod
    def post_calib(self, input_tensor, weight_tensor):
        pass
    
    @abstractmethod
    def pre_compute(self):
        pass

    @abstractmethod
    def post_compute(self):
        """
        计算激活和权重的量化参数（scale/zero等）
        返回：
            dict: {"activation": {...}, "weight": {...}}
        """
        pass
    
    def _init_calibrator(self, param_type: str):
        """初始化校准器（需子类指定激活/权重使用的校准器类型）"""
        calibrator = CalibratorFactory.create(self.strategy.get(param_type).get("calibrator", "minmax"), \
                                              self.strategy.get(param_type))
        return calibrator


class AlgorithmFactory():
    """Algorithm工厂"""
    @staticmethod
    def create(strategy) -> "BaseAlgorithm":
        algo_name = list(strategy["algorithm"].keys())[0]
        if algo_name == 'minmax':
            return MinMax(strategy)
        elif algo_name == 'smoothquant':
            return SmoothQuant(strategy)


class MinMax(BaseAlgorithm):
    def pre_calib(self, input_tensor, weight_tensor):
        pass
    
    def pre_compute(self):
        pass
    
    def post_calib(self, input_tensor, weight_tensor):
        self.act_calibrator.collect(input_tensor)
        self.weight_calibrator.collect(weight_tensor)

    def post_compute(self):
        self.post_compute_info = {
            "activation": self.act_calibrator.compute_params(),
            "weight": self.weight_calibrator.compute_params()
        }
    


class SmoothQuant(BaseAlgorithm):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.pre_act_calibrator = None
        self.pre_weight_calibrator = None
        self.pre_compute_info = {}
        self.alpha = 0.5
        self._setup_smoothquant()
    
    def pre_calib(self, input_tensor, weight_tensor):
        self.pre_act_calibrator.collect(input_tensor)
        self.pre_weight_calibrator.collect(weight_tensor)
    
    def pre_compute(self):
        self._smoothquant()

    def post_calib(self, input_tensor, weight_tensor):
        input_scale = self.pre_compute_info["input"]["scale"]
        weight_scale = self.pre_compute_info["weight"]["scale"]
        processed_input = input_tensor * input_scale
        processed_weight = weight_tensor * weight_scale
        self.act_calibrator.collect(processed_input)
        self.weight_calibrator.collect(processed_weight)
    
    def post_compute(self):
        self.post_compute_info = {
            "activation": self.act_calibrator.compute_params(),
            "weight": self.weight_calibrator.compute_params()
        }

    def _setup_smoothquant(self):
        # 初始化calibrator
        calibrator_type = "minmax"
        act_sub_strategy = self.strategy.get("activation")
        weight_sub_strategy = self.strategy.get("weight")
        act_sub_strategy["granularity"] = "input_channel"
        weight_sub_strategy["granularity"] = "input_channel"
        self.pre_act_calibrator = CalibratorFactory.create(calibrator_type, sub_strategy)
        self.pre_weight_calibrator = CalibratorFactory.create(calibrator_type, sub_strategy)
        # 读取alpha
        algo_params = self.strategy["algorithm"]["smoothquant"]
        self.alpha = algo_params.get("alpha", 0.5)

    def _smoothquant(self):
        act_amax = torch.max(torch.abs(self.pre_act_calibrator.max), torch.abs(self.pre_act_calibrator.min))
        weight_amax = torch.max(torch.abs(self.pre_act_calibrator.max), torch.abs(self.pre_act_calibrator.min))
        eps = 1e-8
        s = (act_amax ** alpha) / ((weight_amax + eps) ** (1 - alpha))
        inv_s = 1.0 / (s + eps)
        self.pre_compute_info = {
            "input": {"scale": inv_s},
            "weight": {"scale": s}
        }
