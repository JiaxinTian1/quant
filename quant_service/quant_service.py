import torch
import torch.nn as nn
import torch.distributed as dist
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


from quant_tool import QuantizerFactory, HookManager
from model_saver import ModelSaverFactory
from model_adapter import ModelAdapterFactory
from .utils import BaseProcessor, DistributedContext, StrategyParser


# --------------------------
# 主控制器
# --------------------------
class QuantService:
    """量化服务主控制器，协调所有处理器"""
    def __init__(self):
        # 初始化分布式上下文
        self.dist_ctx = DistributedContext()
        
        # 核心状态
        self.model_path = None
        self.save_path = None
        self.warpped_model = None
        self.quant_config = None
        self.dataloader = None
        self.use_dist = True
        
        # 量化相关状态
        self.strategy_parser = None
        self.subgraphs = []  # 子图信息（tensor_name列表的列表）
        self.layer_quantizers = {}
        self.quantized_layers = {}  # 已量化的层（tensor_name -> QuantModule）
        self.hook_manager = None
        
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

    def run_pipeline(self, model_path: str, save_path: str, 
                    quant_config_path: str = None, calib_data: List[str] = None) -> None:
        """执行完整量化流程：分离分布式加载与主进程量化步骤"""
        # 第一步：模型加载（所有rank都执行，确保各rank加载对应分片）
        load_params = {
            "model_path": model_path, 
            "save_path": save_path,
            "quant_config_path": quant_config_path
        }
        if self.dist_ctx.is_main_process():
            print("执行步骤: load（所有进程同时执行）")
        # 所有rank共同执行加载步骤（分布式加载）
        self.processors["load"].process(** load_params)
        self.dist_ctx.barrier()  # 等待所有rank加载完成

        # 后续步骤：仅主进程执行核心量化逻辑，其他进程等待
        pipeline_steps = [
            ("init", {"calib_data": calib_data}),
            ("config", {}),
            ("subgraph", {}),
            ("calibration", {}),
            ("quantization", {}),
            ("save", {})
        ]

        for step_name, kwargs in pipeline_steps:
            # 仅主进程执行量化步骤
            if self.dist_ctx.is_main_process():
                print(f"执行步骤: {step_name}（仅主进程执行）")
            self.processors[step_name].process(**kwargs)
            
            # 同步点：确保所有进程在同一步骤对齐
            self.dist_ctx.barrier()

        # 流程结束：仅主进程打印总结
        if self.dist_ctx.is_main_process():
            print("完整量化流程执行完毕")
        
    def __del__(self):
        # 仅在分布式环境已初始化时关闭
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {self.dist_ctx.rank}] 已关闭分布式进程组")


# --------------------------
# 步骤处理器实现
# --------------------------
class LoadProcessor(BaseProcessor):
    """模型加载处理器"""
    def process(self, model_path: str, save_path: str, quant_config_path: str = None, **kwargs) -> None:
        self.pre_process(model_path, quant_config_path)
        
        # 1. 加载量化配置
        if quant_config_path:
            self.quant_service.quant_config = self._load_quant_config(quant_config_path)
            print(f"已从 {quant_config_path} 加载量化配置")
        else:
            print("使用默认量化配置")
        
        # 2. 加载模型、分词器和配置
        print(f"从 {model_path} 加载模型...")
        self.quant_service.model_path = model_path
        self.quant_service.save_path = save_path
        self.quant_service.warpped_model = ModelAdapterFactory.create(
            model_type=self.quant_service.quant_config["model_type"],
            model_path=model_path,
            save_path=save_path,
            dist_ctx=self.quant_service.dist_ctx
        )
        
        self.post_process()

    def _load_quant_config(self, config_path: str) -> Dict:
        """从YAML文件加载量化配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"量化配置文件解析错误: {e}")

    def pre_process(self, model_path: str, quant_config_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        if quant_config_path and not os.path.exists(quant_config_path):
            raise FileNotFoundError(f"量化配置文件不存在: {quant_config_path}")

    def post_process(self):
        print(f"模型加载完成，类型: {type(self.quant_service.warpped_model.model).__name__}")


class InitProcessor(BaseProcessor):
    """初始化处理器"""
    def process(self, calib_data: List[str], **kwargs) -> None:
        self.pre_process()
        
        # 准备模型（切换为eval模式）
        self.quant_service.warpped_model.model.eval()
        
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
            inputs = self.quant_service.warpped_model.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(next(self.quant_service.warpped_model.model.parameters()).device)
            dataloader.append(inputs)
        
        return dataloader

    def pre_process(self):
        save_path = self.quant_service.save_path
        if self.quant_service.dist_ctx.is_main_process():
            # 检查目录是否存在
            if os.path.exists(save_path):
                # 目录存在且非空：提示并删除
                if len(os.listdir(save_path)) > 0:
                    print(f"警告: 保存目录 {save_path} 非空，将覆盖现有文件")
                    shutil.rmtree(save_path)
            else:
                # 目录不存在：创建
                os.makedirs(save_path, exist_ok=True)
                print(f"创建保存目录: {save_path}")

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
        self.quant_service.use_dist = quant_config.get("use_dist", False)
        self.quant_service.quant_type = quant_config.get("quant_type", "fp8")
        self.quant_service.quant_target = quant_config.get("quant_target", "sglang")

    def post_process(self):
        print("配置解析器初始化完成（动态策略解析）")


class SubgraphProcessor(BaseProcessor):
    """子图划分处理器"""
    def process(self, **kwargs) -> None:
        # 过滤可量化层
        self.quant_service.hook_manager = HookManager()
        self._filter_quantizable_layers()
        self.post_process()

    def _filter_quantizable_layers(self) -> List[str]:
        """过滤出可量化的层"""
        state_dict = self.quant_service.warpped_model.model.state_dict()
        
        for tensor_name in state_dict.keys():
            # 只处理权重张量
            
            if not tensor_name.endswith('.weight'):
                continue
            
            strategy = self.quant_service.strategy_parser.get_strategy(
                tensor_name, 
                )
            if not strategy["enable"]:
                continue
            
            # 创建量化器
            layer_path = ".".join(tensor_name.split(".")[:-1])
            layer = self._get_layer_by_path(layer_path)
            # print(f"原始权重设备: {layer.weight.device}")
            quantizer = QuantizerFactory.create(strategy)
            self.quant_service.layer_quantizers[tensor_name] = quantizer
            quantizer.process()
            test_device = torch.device("cuda")
            self.quant_service.hook_manager.register_hook(layer, quantizer)

    def _get_layer_by_path(self, layer_path: str) -> nn.Module:
        """根据路径获取层"""
        module = self.quant_service.warpped_model.model
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
        self._calibrate_subgraph()
        
        self.post_process()

    def _calibrate_subgraph(self):
        """校准单个子图：为每个层创建Algorithm实例并计算参数"""
        
        # 运行校准数据，触发激活值收集
        self._run_calibration_forward()
        self.quant_service.hook_manager.remove_all_hooks()

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
                self.quant_service.warpped_model.forward(batch) # 触发前向传播
            progress_bar.update(1)
        progress_bar.close()

    def post_process(self):
        print(f"校准完成，收集{len(self.quant_service.layer_quantizers)}层量化参数")


class QuantizationProcessor(BaseProcessor):
    """量化处理器：用QuantLinear替换原始层，注入算法计算的参数"""
    def process(self, **kwargs) -> None:
        self.pre_process()

        self._quantize_subgraph(self.quant_service.subgraphs)

        self.post_process()

    def _quantize_subgraph(self, subgraph: List[str]):
        progress_bar = tqdm(
            total=len(self.quant_service.layer_quantizers),
            desc="Quantizing",
            dynamic_ncols=True
        )
        for tensor_name, quantizer in self.quant_service.layer_quantizers.items():
            layer_path = ".".join(tensor_name.split(".")[:-1])
            layer = self._get_layer_by_path(layer_path)
            quantizer = self.quant_service.layer_quantizers[tensor_name]
            # breakpoint()
            quantizer.quantize(layer)
            progress_bar.update(1)
        progress_bar.close()

    def post_process(self):
        print(f"量化完成")
    
    def _get_layer_by_path(self, layer_path: str) -> nn.Module:
        """根据路径获取层"""
        module = self.quant_service.warpped_model.model
        for part in layer_path.split("."):
            module = getattr(module, part)
        return module


class SaveProcessor(BaseProcessor):
    """保存处理器（适配safetensors格式）"""
    
    def process(self, **kwargs) -> None:
        self.pre_process()
        
        saver = ModelSaverFactory.create(self.quant_service)
        saver.save()
        
        self.post_process()


    def pre_process(self):
        pass


    def post_process(self):
        print(f"量化模型已保存至 {self.quant_service.save_path}")


