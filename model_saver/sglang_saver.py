import torch
import torch.nn as nn
import os
import json
import math
import shutil
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from safetensors.torch import save_file
from .base import BaseSaver


class SglangSaver(BaseSaver):
    def __init__(self, quant_service):
        super().__init__(quant_service)
        self.weight_map = {}
        self.max_file_size_bytes = int(5 * 1024 ** 3)
        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)
    
    def save(self):
        model_state_dict = self._process_state_dict(self.warpped_model.model.state_dict())
        # 分离 act scale
        self.save_act_params(model_state_dict)
        # 分块保存模型权重
        self.save_state_dict_with_index(model_state_dict)
        self.save_weight_map()
        # 保存配置、分词器和量化参数
        self.warpped_model.copy_files()
        self.save_quantization_config()
        self.save_hf_quant_config()
        
        # self.quant_service.tokenizer.save_pretrained(self.save_path)
    
    def _process_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """收集当前进程的state_dict"""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = self.warpped_model.mapping_tensor_params(key)
            new_state_dict[new_key] = value
        return new_state_dict


    def save_act_params(self, state_dict: Dict[str, torch.Tensor]):
        act_params = {}
        keys_to_process = []
        act_filename = "input_scales.safetensors"
        for key, value in state_dict.items():
            if "input_scale" in key:
                act_params[key] = value
                self.weight_map[key] = act_filename
        if act_params:
            act_path = os.path.join(self.save_path, act_filename)
            save_file(act_params, act_path)

    
    def save_state_dict_with_index(self, state_dict: Dict[str, torch.Tensor]) -> dict:
        """合并保存和索引生成，避免两次遍历"""
        current_size = 0
        current_chunk = {}
        chunk_files = []
        chunk_idx = 1
        total_size = 0

        estimated_num_files = self._estimate_num_files(state_dict)

        progress_bar = tqdm(
            total=estimated_num_files,
            desc="Saving model chunks",
            dynamic_ncols=True
        )
        
        for name, tensor in state_dict.items():
            if "input_scale" in name:
                continue
            tensor_size = tensor.numel() * tensor.element_size()
            total_size += tensor_size
            
            # 如果当前块加上新张量会超过限制，且当前块不为空，则保存当前块
            if current_size + tensor_size > self.max_file_size_bytes and current_chunk:
                filename = f"model-{chunk_idx:05d}-of-{estimated_num_files:05d}.safetensors"
                filepath = os.path.join(self.save_path, filename)
                save_file(current_chunk, filepath)
                chunk_files.append(filename)
                current_chunk = {}
                current_size = 0
                chunk_idx += 1

                progress_bar.update(1)
            
            # 添加张量到当前块并记录映射关系
            current_filename = f"model-{chunk_idx:05d}-of-{estimated_num_files:05d}.safetensors"
            current_chunk[name] = tensor
            self.weight_map[name] = current_filename
            current_size += tensor_size
        
        # 保存最后一个块
        if current_chunk:
            filename = f"model-{chunk_idx:05d}-of-{estimated_num_files:05d}.safetensors"
            filepath = os.path.join(self.save_path, filename)
            save_file(current_chunk, filepath)
            chunk_files.append(filename)
            progress_bar.update(1)
        progress_bar.close()
    
    def _estimate_num_files(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """预估最终会生成多少个safetensors文件"""
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        return math.ceil(total_size / self.max_file_size_bytes)
    
    def save_weight_map(self):
        # 保存索引文件
        index_info = {"weight_map": self.weight_map}
        index_path = os.path.join(self.save_path, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_info, f, indent=2, ensure_ascii=False)

    @abstractmethod
    def save_quantization_config(self):
        pass
    
    @abstractmethod
    def save_hf_quant_config(self):
        pass
    
    


class SglangFP8Saver(SglangSaver):
    def __init__(self, quant_service):
        super().__init__(quant_service)
    
    
    def save_quantization_config(self):
        # 1. 读取原始 config.json
        config_path = os.path.join(self.model_path, "config.json")
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

    def save_hf_quant_config(self):
        """保存量化参数（保持原有逻辑，无需修改）"""
        pass


class SglangIN4Saver(SglangSaver):
    def __init__(self, quant_service):
        super().__init__(quant_service)
    
    
    def save_quantization_config(self):
        # 1. 读取原始 config.json
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config: Dict[str, Any] = json.load(f)
        
        # 2. 添加/更新量化配置
        del config["quantization_config"]
    
        # 3. 保存到新路径
        with open(os.path.join(self.save_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save_hf_quant_config(self):
        """保存量化参数（保持原有逻辑，无需修改）"""
        hf_quant_config = {}
        hf_quant_config['quantization'] = {}
        hf_quant_config['quantization']["quant_algo"] = "MIXED_PRECISION"
        hf_quant_config['quantization']["kv_cache_quant_algo"] = None
        with open(os.path.join(self.save_path, "hf_quant_config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_quant_config, f, indent=2, ensure_ascii=False)