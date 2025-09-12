import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import yaml
from copy import deepcopy
import torch.distributed as dist


class BaseProcessor:
    """处理器基类，所有步骤处理器继承此类"""
    def __init__(self, quant_service: "QuantService"):
        self.quant_service = quant_service
        self.state = {}

    def pre_process(self, **kwargs) -> None:
        """处理前的准备工作"""
        pass

    def process(self, **kwargs) -> None:
        """核心处理逻辑"""
        raise NotImplementedError

    def post_process(self, **kwargs) -> None:
        """处理后的清理工作"""
        pass


# --------------------------
# 辅助类
# --------------------------

class DistributedContext:
    """分布式环境上下文（修复device_id类型错误）"""
    def __init__(self):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
        # 1. 先确定当前进程的设备（返回torch.device对象，而非整数）
        self.device = self._get_device()
        
        # 2. 绑定设备（确保当前进程使用指定GPU）
        # if self.device.type == "cuda":
        #     torch.cuda.set_device(self.device)
        
        # 3. 初始化分布式进程组（传入正确类型的device_id）
        self.init_process_group()

    def _get_device(self) -> torch.device:
        """获取当前进程的设备（返回torch.device对象）"""
        if self.world_size > 1 and torch.cuda.is_available():
            # 多卡场景：设备为 cuda:local_rank（如cuda:6）
            return torch.device(f"cuda:{self.local_rank}")
        elif torch.cuda.is_available():
            # 单卡场景：默认cuda设备
            return torch.device("cuda")
        else:
            # CPU场景
            return torch.device("cpu")

    def init_process_group(self):
        """初始化分布式进程组（关键：device_id传入torch.device对象）"""
        if self.world_size > 1 and not dist.is_initialized():
            # 核心修改：device_id使用self.device（torch.device对象），而非整数
            init_kwargs = {
                "backend": "nccl",
                "world_size": self.world_size,
                "rank": self.rank,
                # 正确类型：torch.device对象（支持访问.index属性）
                "device_id": self.device if self.device.type == "cuda" else None
            }

            # 单机多卡默认使用env://初始化（依赖WORLD_SIZE/RANK/LOCAL_RANK环境变量）
            dist.init_process_group(**init_kwargs)
            print(f"[Rank {self.rank}] 分布式进程组初始化完成，绑定设备: {self.device}")
    
    def barrier(self):
        """进程同步屏障，仅在分布式环境有效"""
        if self.world_size > 1:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """判断当前进程是否为主进程（rank=0）"""
        return self.rank == 0


class StrategyParser:
    """动态策略解析器"""
    def __init__(self, quant_config: dict):
        self.quant_config = quant_config
        self.global_cfg = self.quant_config.get("global", {})
        self.local_cfg = self.quant_config.get("local", {})
        self.disable_patterns = self.quant_config.get("disable", [])
        self._cache = {}

    def get_strategy(self, tensor_name: str) -> Dict[str, Any]:
        """动态解析单个层的策略"""
        if tensor_name in self._cache:
            return self._cache[tensor_name]
        
        # 检查是否禁用
        if self._is_disabled(tensor_name):
            strategy = {"enable": False}
        else:
            # 基础策略（全局配置）
            strategy = {
                "enable": False,
                "need_preprocess": False,
                "tensor_name": tensor_name,
                "quant_type": self.quant_config.get("quant_type", "fp8"),
                "weight": deepcopy(self.global_cfg.get("weight", {})),
                "activation": deepcopy(self.global_cfg.get("activation", {})),
                "algorithm": deepcopy(self.global_cfg.get("algorithm", {})),
            }
            strategy["weight"]["original_dtype"] = self.quant_config["original_dtype"]
            strategy["activation"]["original_dtype"] = self.quant_config["original_dtype"]

            # 应用局部配置覆盖
            self._apply_local_overrides(tensor_name, strategy)
            
            if strategy["weight"]["enable"] or strategy["activation"]["enable"]:
                strategy["enable"] = True
            for algo in strategy["activation"].keys():
                if algo in ["smoothquant"]:
                    strategy["need_preprocess"] = True
        
        self._cache[tensor_name] = strategy
        return strategy

    def _is_disabled(self, tensor_name: str) -> bool:
        """判断当前层是否被禁用"""
        return any(pattern in tensor_name for pattern in self.disable_patterns)

    def _apply_local_overrides(self, tensor_name: str, strategy: Dict[str, Any]) -> None:
        """用局部配置覆盖全局配置"""
        if self.local_cfg is None:
            return
        for pattern, local_settings in self.local_cfg.items():
            if pattern in tensor_name:
                if "weight" in local_settings:
                    strategy["weight"].update(local_settings["weight"])
                if "activation" in local_settings:
                    strategy["activation"].update(local_settings["activation"])
                if "algorithm" in local_settings:
                    strategy["algorithm"].update(local_settings["algorithm"])