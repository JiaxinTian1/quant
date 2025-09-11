import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class BaseAlgorithm(ABC):
    """量化算法策略接口，所有具体算法（SmoothQuant/AWQ）需实现此类"""
    def __init__(self, strategy):
        self.strategy = strategy
    
    @abstractmethod
    def compute_params(self, act_calibrator, weight_calibrator, strategy):
        """
        计算激活和权重的量化参数（scale/zero等）
        参数：
            act_calibrator: 激活校准器（含激活统计数据）
            weight_calibrator: 权重校准器（含权重统计数据）
            strategy: 量化配置（如alpha/balance等超参）
        返回：
            dict: {"activation": {...}, "weight": {...}}
        """
        pass


class QuantAlgorithmFactory():
    """Algorithm工厂"""
    @staticmethod
    def create(algorithm, params) -> "BaseAlgorithm":
        if algorithm == 'minmax':
            return MinMax(params)
        elif algorithm == 'smoothquant':
            return SmoothQuant(params)


class MinMax(BaseAlgorithm):
    def compute_params(self, act_calibrator, weight_calibrator):
        return {
            "activation": act_calibrator.compute_params(),
            "weight": weight_calibrator.compute_params()
        }
    


class SmoothQuant(BaseAlgorithm):
    def compute_params(self, act_calibrator, weight_calibrator):
        pass
