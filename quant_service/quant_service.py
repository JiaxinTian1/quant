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
from dataclasses import dataclass

from quant_tool import QuantizerFactory, HookManager
from model_saver import ModelSaverFactory
from model_adapter import ModelAdapterFactory
from quant_practice import get_default_config_practice, get_default_data_practice
from .utils import BaseProcessor, DistributedContext, StrategyParser


# --------------------------
# 主控制器
# --------------------------
@dataclass
class QuantParams:
    model_path: str
    save_path: str
    quant_type: str
    quant_target: str
    quant_config_path: Optional[str] = None
    calib_data_path: Optional[str] = None


class QuantService:
    """量化服务主控制器，协调所有处理器"""
    def __init__(self):
        # 初始化分布式上下文
        self.dist_ctx = DistributedContext()
        
        # 核心状态
        self.quant_params = None

        self.warpped_model = None
        self.quant_config = {}
        self.dataloader = None
        
        # 量化相关状态
        self.strategy_parser = None
        self.subgraphs = []  # 子图信息（tensor_name列表的列表）
        self.layer_quantizers = {}
        self.quantized_layers = {}  # 已量化的层（tensor_name -> QuantModule）
        self.hook_manager = None
        self.need_preprocess = False
        
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
        self.pre_steps = [
            ("load", {}),
            ("config", {}),
            ("init", {}),
            ("subgraph", {}),
        ]

        self.quant_steps = [
            ("calibration", {}),
            ("quantization", {}),
        ]
        self.post_steps = [("save", {})]
    
    def service(self):
        self.run_pipeline(self.pre_steps)
        self.run_pipeline(self.quant_steps)
        self.run_pipeline(self.post_steps)

    def run_pipeline(self, pipeline_steps) -> None:
        """执行完整量化流程：分离分布式加载与主进程量化步骤"""

        # 后续步骤：仅主进程执行核心量化逻辑，其他进程等待
        for step_name, kwargs in pipeline_steps:
            # 仅主进程执行量化步骤
            if self.dist_ctx.is_main_process():
                print(f"执行步骤: {step_name}（仅主进程执行）")
            self.processors[step_name].process(**kwargs)
            
            # 同步点：确保所有进程在同一步骤对齐
            self.dist_ctx.barrier()

    
    def build_quant_pipline(self):
        pass

    def __del__(self):
        # 流程结束：仅主进程打印总结
        if self.dist_ctx.is_main_process():
            print("完整量化流程执行完毕")
        # 仅在分布式环境已初始化时关闭
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {self.dist_ctx.rank}] 已关闭分布式进程组")


# --------------------------
# 步骤处理器实现
# --------------------------
class LoadProcessor(BaseProcessor):
    """模型加载处理器"""
    def process(self, **kwargs) -> None:
        model_path = self.quant_service.quant_params.model_path
        save_path = self.quant_service.quant_params.save_path
        self.pre_process(model_path)
        
        # 加载模型、分词器和配置
        print(f"从 {model_path} 加载模型...")
        self.quant_service.model_path = model_path
        self.quant_service.save_path = save_path
        self.quant_service.warpped_model = ModelAdapterFactory.create(
            model_type=self.quant_service.model_type,
            model_path=model_path,
            save_path=save_path,
            dist_ctx=self.quant_service.dist_ctx
        )
        
        self.post_process()


    def _parse_model_config(self, model_path):
        original_config_path = os.path.join(model_path, "config.json")
        with open(original_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.quant_service.model_type = config['model_type']
        self.quant_service.original_dtype = config['torch_dtype']

    def pre_process(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        self._parse_model_config(model_path)

    def post_process(self):
        print(f"模型加载完成，类型: {type(self.quant_service.warpped_model.model).__name__}")


class ConfigProcessor(BaseProcessor):
    """配置处理器，初始化动态策略解析器"""
    def process(self, **kwargs) -> None:
        quant_config_path = self.quant_service.quant_params.quant_config_path

        self.pre_process()
        if quant_config_path:
            self.quant_service.quant_config = self._load_quant_config(quant_config_path)
            print(f"已从 {quant_config_path} 加载量化配置")
        else:
            self.quant_service.quant_config = get_default_config_practice(
                self.quant_service.model_type,
                self.quant_service.quant_params.quant_type,
            )
            print("使用默认量化配置")
        
        self.quant_service.quant_config["quant_type"] = self.quant_service.quant_params.quant_type
        self.quant_service.quant_config["original_dtype"] = self.quant_service.original_dtype
        
        self.post_process()
    
    def _load_quant_config(self, config_path: str) -> Dict:
        """从YAML文件加载量化配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"量化配置文件解析错误: {e}")

    def post_process(self):
        # 初始化动态策略解析器
        self.quant_service.strategy_parser = StrategyParser(
            quant_config=self.quant_service.quant_config
        )
        print("配置解析器初始化完成（动态策略解析）")


class InitProcessor(BaseProcessor):
    """初始化处理器"""
    def process(self, **kwargs) -> None:
        self.pre_process()
        
        # 准备模型（切换为eval模式）
        self.quant_service.warpped_model.model.eval()
        
        # 准备校准数据加载器
        self.quant_service.dataloader = self._create_calib_dataloader()
        
        self.post_process()

    def _create_calib_dataloader(self, batch_size: int = 2) -> List[Dict]:
        """创建校准数据加载器"""
        calib_data = []
        calib_data_path = self.quant_service.quant_params.calib_data_path
        if calib_data_path:
            calib_data = self._load_calib_data(calib_data_path)
        else:
            calib_data = get_default_data_practice()
            print("使用默认校准集")
        
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
    
    def _load_calib_data(self, data_path: str) -> Dict:
        """从JSON文件加载校准数据"""
        with open(data_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"校准数据JSON文件解析错误: {e}")
            except IOError as e:
                raise ValueError(f"读取校准数据文件失败: {e}")

    def pre_process(self):
        save_path = self.quant_service.quant_params.save_path
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
            self.quant_service.hook_manager.register_hook(layer, quantizer)
            if strategy["need_preprocess"]:
                self.quant_service.need_preprocess = True
                quantizer.need_preprocess = True
                quantizer.quant_status = "pre_calib"

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
        if self.quant_service.need_preprocess:
            self._preprocess_subgraph()
        self._calibrate_subgraph()
        
        self.post_process()
    
    def _preprocess_subgraph(self):
        """预处理"""
        self._run_calibration_forward()
        for tensor_name, quantizer in self.quant_service.layer_quantizers.items():
            quantizer.pre_compute()
            quantizer.quant_status = "post_calib"

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
        print(f"量化模型已保存至 {self.quant_service.quant_params.save_path}")


