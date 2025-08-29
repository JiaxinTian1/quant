import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import yaml
import fnmatch
import shutil
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig, AutoModelForCausalLM
from copy import deepcopy
from abc import ABC, abstractmethod
from safetensors.torch import save_file
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


from .quantizer import QuantizerFactory
from .hook import HookManager
from .saver import SaverFactory


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
# 主控制器
# --------------------------
class QuantService:
    """量化服务主控制器，协调所有处理器"""
    def __init__(self):
        # 核心状态
        self.model_path = None
        self.save_path = None
        self.model = None
        self.tokenizer = None
        self.config = None
        self.quant_config = None
        self.dataloader = None
        
        # 量化相关状态
        self.strategy_parser = None
        self.subgraphs = []  # 子图信息（tensor_name列表的列表）
        self.layer_quantizers = {}
        self.quant_params = {}  # 量化参数（tensor_name -> 参数字典）
        self.quantized_layers = {}  # 已量化的层（tensor_name -> QuantModule）
        
        # 类型信息
        self.original_dtype = None
        
        # 初始化所有处理器
        self.processors = {
            "load": LoadProcessor(self),
            "init": InitProcessor(self),
            "config": ConfigProcessor(self),
            "subgraph": SubgraphProcessor(self),
            "calibration": CalibrationProcessor(self),
            "quantization": QuantizationProcessor(self),
            "save": SaveProcessor(self)
        }

    def run_pipeline(self, model_path: str, save_dir: str, 
                    quant_config_path: str = None, calib_data: List[str] = None) -> None:
        """执行完整量化流程"""
        pipeline_steps = [
            ("load", {"model_path": model_path, "quant_config_path": quant_config_path}),
            ("init", {"calib_data": calib_data}),
            ("config", {}),
            ("subgraph", {}),
            ("calibration", {}),
            ("quantization", {}),
            ("save", {"save_dir": save_dir})
        ]
        
        for step_name, kwargs in pipeline_steps:
            print(f"执行步骤: {step_name}")
            self.processors[step_name].process(**kwargs)


# --------------------------
# 步骤处理器实现
# --------------------------
class LoadProcessor(BaseProcessor):
    """模型加载处理器"""
    def process(self, model_path: str, quant_config_path: str = None, **kwargs) -> None:
        self.pre_process(model_path, quant_config_path)
        
        # 1. 加载量化配置
        if quant_config_path:
            self.quant_service.quant_config = self._load_quant_config(quant_config_path)
            print(f"已从 {quant_config_path} 加载量化配置")
        else:
            self.quant_service.quant_config = self._get_default_quant_config()
            print("使用默认量化配置")
        
        # 2. 加载模型、分词器和配置
        print(f"从 {model_path} 加载模型...")
        self.quant_service.model_path = model_path
        self.quant_service.config = AutoConfig.from_pretrained(
            model_path,
            config = self.quant_service.config,
            trust_remote_code=True
        )
        self.quant_service.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            config = self.quant_service.config,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.quant_service.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        
        self.post_process()

    def _load_quant_config(self, config_path: str) -> Dict:
        """从YAML文件加载量化配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"量化配置文件解析错误: {e}")

    def _get_default_quant_config(self) -> Dict:
        """默认量化配置"""
        return {
            "global": {
                "weight": {
                    "quant_dtype": "int8",
                    "calibrator": "minmax",
                    "algorithm": "minmax",
                    "granularity": "channel"
                },
                "activation": {
                    "quant_dtype": "int8",
                    "calibrator": "minmax",
                    "algorithm": "minmax",
                    "granularity": "token"
                }
            },
            "max_layers_per_subgraph": 5,
            "is_training": False,
            "disable": []
        }

    def pre_process(self, model_path: str, quant_config_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        if quant_config_path and not os.path.exists(quant_config_path):
            raise FileNotFoundError(f"量化配置文件不存在: {quant_config_path}")

    def post_process(self):
        print(f"模型加载完成，类型: {type(self.quant_service.model).__name__}")


class InitProcessor(BaseProcessor):
    """初始化处理器"""
    def process(self, calib_data: List[str], **kwargs) -> None:
        self.pre_process()
        
        # 准备模型（切换为eval模式）
        self.quant_service.model = self.quant_service.model.eval()
        # self.quant_service.model.to('cpu') 
        
        # 准备校准数据加载器
        self.quant_service.dataloader = self._create_calib_dataloader(calib_data)
        
        self.post_process()

    def _create_calib_dataloader(self, calib_data: List[str], batch_size: int = 2) -> List[Dict]:
        """创建校准数据加载器"""
        if not calib_data:
            calib_data = [
                "这是一个用于模型量化的校准样本。",
                "量化可以显著减少模型大小并加速推理。",
                "Transformer模型通常包含多个注意力层和前馈网络。"
            ]
        
        dataloader = []
        for i in range(0, len(calib_data), batch_size):
            batch_text = calib_data[i:i+batch_size]
            inputs = self.quant_service.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.quant_service.model.device)
            dataloader.append(inputs)
        
        return dataloader

    def post_process(self):
        print(f"初始化完成，校准数据批次: {len(self.quant_service.dataloader)}")


class ConfigProcessor(BaseProcessor):
    """配置处理器，初始化动态策略解析器"""
    def process(self, **kwargs) -> None:
        self.pre_process()
        
        # 初始化动态策略解析器
        self.quant_service.strategy_parser = StrategyParser(
            quant_config=self.quant_service.quant_config
        )
        
        # 解析通用配置
        self._parse_general_configs()
        
        self.post_process()

    def _parse_general_configs(self) -> None:
        """解析非层相关的配置"""
        quant_config = self.quant_service.quant_config
        self.quant_service.max_layers_per_subgraph = quant_config.get(
            "max_layers_per_subgraph", 5
        )
        self.quant_service.is_training = quant_config.get("is_training", False)
        self.quant_service.quant_type = quant_config.get("quant_type", "fp8")
        self.quant_service.quant_target = quant_config.get("quant_target", "sglang")

    def post_process(self):
        print("配置解析器初始化完成（动态策略解析）")


class SubgraphProcessor(BaseProcessor):
    """子图划分处理器"""
    def process(self, **kwargs) -> None:
        # 过滤可量化层
        quantizable_layers = self._filter_quantizable_layers()
        
        # 按最大层数划分子图
        max_layers = self.quant_service.quant_config.get("max_layers_per_subgraph", 5)

        max_layers = len(quantizable_layers)

        self.quant_service.subgraphs = [
            quantizable_layers[i:i+max_layers] 
            for i in range(0, len(quantizable_layers), max_layers)
        ]
        self.post_process()

    def _filter_quantizable_layers(self) -> List[str]:
        """过滤出可量化的层"""
        disable_patterns = self.quant_service.quant_config.get("disable", [])
        quantizable = []
        state_dict = self.quant_service.model.state_dict()
        
        for tensor_name in state_dict.keys():
            # 只处理权重张量
            if not tensor_name.endswith('.weight'):
                continue
                
            # 只处理Linear层
            layer_path = ".".join(tensor_name.split(".")[:-1])
            try:
                layer = self._get_layer_by_path(layer_path)
                if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                    continue
            except:
                continue
                
            # 检查是否被禁用
            if any(fnmatch.fnmatch(tensor_name, pattern) for pattern in disable_patterns):
                continue 
            quantizable.append(tensor_name)
        return quantizable

    def _get_layer_by_path(self, layer_path: str) -> nn.Module:
        """根据路径获取层"""
        module = self.quant_service.model
        for part in layer_path.split("."):
            module = getattr(module, part)
        return module

    def post_process(self):
        total_layers = sum(len(sg) for sg in self.quant_service.subgraphs)
        print(f"子图划分完成，{len(self.quant_service.subgraphs)}个子图，共{total_layers}层")


class CalibrationProcessor(BaseProcessor):

    def __init__(self, quant_service):
        super().__init__(quant_service)
        self.hook_manager = None
    
    """校准处理器"""
    def process(self, **kwargs) -> None:
        self.pre_process()
        self.hook_manager = HookManager()
        for i, subgraph in enumerate(self.quant_service.subgraphs):
            print(f"校准子图 {i+1}/{len(self.quant_service.subgraphs)}")
            self._calibrate_subgraph(subgraph)
        
        self.post_process()

    def _calibrate_subgraph(self, subgraph: List[str]):
        """校准单个子图：为每个层创建Algorithm实例并计算参数"""
        # 为子图中的每个层创建算法实例
        for tensor_name in subgraph:
            strategy = self.quant_service.strategy_parser.get_strategy(
                tensor_name, 
                )
            if not strategy["enabled"]:
                continue
            # 创建量化器
            layer_path = ".".join(tensor_name.split(".")[:-1])
            layer = self._get_layer_by_path(layer_path)
            # print(f"原始权重设备: {layer.weight.device}")
            quantizer = QuantizerFactory.create(strategy)
            self.quant_service.layer_quantizers[tensor_name] = quantizer
            quantizer.process()
            test_device = torch.device("cuda")
            self.hook_manager.register_hook(layer, quantizer)
        
        # 运行校准数据，触发激活值收集
        self._run_calibration_forward()


    def _get_layer_by_path(self, layer_path: str) -> nn.Module:
        """根据路径获取层模块"""
        module = self.quant_service.model
        for part in layer_path.split("."):
            module = getattr(module, part)
        return module

    def _run_calibration_forward(self):
        """运行校准数据，触发Hook收集激活值"""
        progress_bar = tqdm(
            total=len(self.quant_service.dataloader),
            desc="Calibrating",
            dynamic_ncols=True
        )
        for batch in self.quant_service.dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                self.quant_service.model(** batch)  # 触发前向传播
            progress_bar.update(1)
        progress_bar.close()

    def post_process(self):
        self.hook_manager.remove_all_hooks()
        print(f"校准完成，收集{len(self.quant_service.layer_quantizers)}层量化参数")


class QuantizationProcessor(BaseProcessor):
    """量化处理器：用QuantLinear替换原始层，注入算法计算的参数"""
    def process(self, **kwargs) -> None:
        self.pre_process()

        for sg_idx, subgraph in enumerate(self.quant_service.subgraphs):
            print(f"量化子图 {sg_idx+1}/{len(self.quant_service.subgraphs)}")
            self._quantize_subgraph(subgraph)

        self.post_process()

    def _quantize_subgraph(self, subgraph: List[str]):
        progress_bar = tqdm(
            total=len(self.quant_service.layer_quantizers),
            desc="Quantizing",
            dynamic_ncols=True
        )
        for tensor_name in subgraph:
            if tensor_name not in self.quant_service.layer_quantizers:
                continue
            layer_path = ".".join(tensor_name.split(".")[:-1])
            layer = self._get_layer_by_path(layer_path)
            quantizer = self.quant_service.layer_quantizers[tensor_name]
            quantizer.quantize(layer)
            progress_bar.update(1)
        progress_bar.close()

    def post_process(self):
        print(f"量化完成")
    
    def _get_layer_by_path(self, layer_path: str) -> nn.Module:
        """根据路径获取层"""
        module = self.quant_service.model
        for part in layer_path.split("."):
            module = getattr(module, part)
        return module


class SaveProcessor(BaseProcessor):
    """保存处理器（适配safetensors格式）"""
    def process(self, save_dir: str, **kwargs) -> None:
        self.pre_process(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.quant_service.save_path = save_dir
        
        saver = SaverFactory.create(self.quant_service)
        saver.save()
        
        self.post_process(save_dir)


    def pre_process(self, save_dir: str):
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            print(f"警告: 保存目录 {save_dir} 非空，可能会覆盖现有文件")
            shutil.rmtree(save_dir)


    def post_process(self, save_dir: str):
        print(f"量化模型已保存至 {save_dir}")


# --------------------------
# 辅助类
# --------------------------
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
            strategy = {"enabled": False}
        else:
            # 基础策略（全局配置）
            strategy = {
                "enabled": True,
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
        
        self._cache[tensor_name] = strategy
        return strategy

    def _is_disabled(self, tensor_name: str) -> bool:
        """判断当前层是否被禁用"""
        return any(pattern in tensor_name for pattern in self.disable_patterns)

    def _apply_local_overrides(self, tensor_name: str, strategy: Dict[str, Any]) -> None:
        """用局部配置覆盖全局配置"""
        for pattern, local_settings in self.local_cfg.items():
            if pattern in tensor_name:
                if "weight" in local_settings:
                    strategy["weight"].update(local_settings["weight"])
                if "activation" in local_settings:
                    strategy["activation"].update(local_settings["activation"])
                if "algorithm" in local_settings:
                    strategy["algorithm"].update(local_settings["algorithm"])
