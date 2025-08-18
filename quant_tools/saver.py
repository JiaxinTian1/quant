import torch
import torch.nn as nn
import os
import json
import math
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from safetensors.torch import save_file

class BaseSaver(ABC):
    def __init__(self, quant_service):
        self.quant_service = quant_service
        self.model_path = quant_service.model_path
        self.save_path = quant_service.save_path
        self.model = quant_service.model
        self.max_file_size_bytes = int(5 * 1024 ** 3)
    
    @abstractmethod
    def save(self):
        pass


class SaverFactory:
    """保存器工厂"""
    @staticmethod
    def create(quant_service) -> "BaseSaver":
        if quant_service.quant_type == "fp8_e4m3" and quant_service.quant_target == "sglang":
            return SglangFP8Saver(quant_service)

class SglangFP8Saver(BaseSaver):
    def __init__(self, quant_service):
        super().__init__(quant_service)
    
    def save(self):
        model_state_dict = self.model.state_dict()
        # 分块保存模型权重
        safetensors_files = self._save_state_dict_with_index(model_state_dict)
        # 保存配置、分词器和量化参数
        self.quant_service.config.save_pretrained(self.save_path)
        self.quant_service.tokenizer.save_pretrained(self.save_path)
        self._save_quantization_config()
        self._save_hf_quant_config()

    
    def _save_state_dict_with_index(self, state_dict: Dict[str, torch.Tensor]) -> dict:
        """合并保存和索引生成，避免两次遍历"""
        current_size = 0
        current_chunk = {}
        weight_map = {}
        chunk_files = []
        chunk_idx = 1
        total_size = 0

        estimated_num_files = self._estimate_num_files(state_dict)
        
        for name, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            total_size += tensor_size
            filename = f"model-{chunk_idx:05d}-of-{estimated_num_files:05d}.safetensors"
            # 如果当前块加上新张量会超过限制，且当前块不为空，则保存当前块
            if current_size + tensor_size > self.max_file_size_bytes and current_chunk:
                filepath = os.path.join(self.save_path, filename)
                save_file(current_chunk, filepath)
                chunk_files.append(filename)
                current_chunk = {}
                current_size = 0
                chunk_idx += 1
            
            # 添加张量到当前块并记录映射关系
            current_chunk[name] = tensor
            weight_map[name] = filename
            current_size += tensor_size
        
        # 保存最后一个块
        if current_chunk:
            filename = f"model-{chunk_idx:05d}-of-{estimated_num_files:05d}.safetensors"
            filepath = os.path.join(self.save_path, filename)
            save_file(current_chunk, filepath)
            chunk_files.append(filename)
        
        index_info = {
            "metadata": {
                "total_size": total_size,
                "max_file_size": self.max_file_size_bytes,
                "num_files": len(chunk_files)
            },
            "weight_map": weight_map
        }
        
        # 保存索引文件
        index_path = os.path.join(self.save_path, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_info, f, indent=2, ensure_ascii=False)
    
    def _estimate_num_files(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """预估最终会生成多少个safetensors文件"""
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        return math.ceil(total_size / self.max_file_size_bytes)
    
    def _save_quantization_config(self):
        # 1. 读取原始 config.json
        config_path = os.path.join(self.save_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config: Dict[str, Any] = json.load(f)
        
        # 2. 添加/更新量化配置
        config["quantization_config"] = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        }
    
        # 3. 保存到新路径
        with open(os.path.join(self.save_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _save_hf_quant_config(self):
        """保存量化参数（保持原有逻辑，无需修改）"""
        hf_quant_config = {}
        hf_quant_config['quantization'] = {}
        hf_quant_config['quantization']["quant_algo"] = "MIXED_PRECISION"
        hf_quant_config['quantization']["kv_cache_quant_algo"] = None
        with open(os.path.join(self.save_path, "hf_quant_config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_quant_config, f, indent=2, ensure_ascii=False)