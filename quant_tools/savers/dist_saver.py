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


class DistSaver(BaseSaver):
    def __init__(self, quant_service):
        super().__init__(quant_service)
        self.dist_ctx = quant_service.dist_ctx
        self.rank = self.dist_ctx.rank
        self.world_size = self.dist_ctx.world_size
        self.save_path = quant_service.save_path  # 直接使用最终保存目录
        
        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)

    def save(self):
        # 1. 所有进程：直接保存自己的分片和局部映射到最终目录
        self._save_rank_files()
        
        # 同步：等待所有进程完成写入
        self.dist_ctx.barrier()
        
        # 2. 仅主进程：合并局部映射生成完整索引，处理共享文件
        if self.dist_ctx.is_main_process():
            self._merge_local_maps()
            self._merge_act_params()
            self.save_quantization_config()
            self.save_hf_quant_config()
            self.copy_files()
            self._cleanup_local_files()  # 清理局部映射文件
            print(f"[Rank 0] 保存完成，目录: {self.save_path}")

    def _save_rank_files(self):
        """当前进程直接保存分片和局部映射到最终目录"""
        model_state_dict = self.model.state_dict()
        
        # 1. 保存权重分片（文件名含rank，确保唯一）
        chunk_id = self.rank + 1
        total_chunks = self.world_size
        chunk_filename = f"model-{chunk_id:05d}-of-{total_chunks:05d}.safetensors"
        chunk_path = os.path.join(self.save_path, chunk_filename)
        save_file(model_state_dict, chunk_path)
        print(f"[Rank {self.rank}] 已保存分片: {chunk_filename}")
        
        # 2. 保存当前进程的act参数（文件名含rank，避免冲突）
        act_params = self._collect_act_params(model_state_dict)
        act_filename = None
        if act_params:
            act_filename = f"input_scales_rank{self.rank}.safetensors"
            act_path = os.path.join(self.save_path, act_filename)
            save_file(act_params, act_path)
        
        # 3. 保存局部权重映射（记录当前进程的权重→文件名对应关系）
        self._save_local_map(chunk_filename, act_filename)

    def _collect_act_params(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """收集当前进程的act参数"""
        act_params = {}
        for key, value in state_dict.items():
            if "input_scale" in key:
                new_key = self._mapping_act_params(key)
                act_params[new_key] = value
        return act_params

    def _save_local_map(self, chunk_filename: str, act_filename: str):
        """保存当前进程的局部权重映射（供主进程合并）"""
        local_map = {}
        # 记录权重映射
        for name in self.model.state_dict().keys():
            local_map[name] = chunk_filename
        
        # 记录act参数映射（若有）
        if act_filename:
            act_params = self._collect_act_params(self.model.state_dict())
            for key in act_params.keys():
                local_map[key] = "input_scales.safetensors"  # 最终合并后的文件名
        
        # 保存局部映射（文件名含rank，避免冲突）
        map_path = os.path.join(self.save_path, f"local_map_rank{self.rank}.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(local_map, f, indent=2)

    def _merge_local_maps(self):
        """主进程合并所有局部映射，生成完整索引"""
        final_weight_map = {}
        for rank in range(self.world_size):
            map_path = os.path.join(self.save_path, f"local_map_rank{rank}.json")
            if not os.path.exists(map_path):
                continue  # 跳过无映射的进程（理论上不会出现）
            
            with open(map_path, "r", encoding="utf-8") as f:
                rank_map = json.load(f)
                final_weight_map.update(rank_map)
        
        # 保存完整索引
        index_path = os.path.join(self.save_path, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"weight_map": final_weight_map}, f, indent=2)
        print(f"[Rank 0] 已生成完整索引，共 {len(final_weight_map)} 个条目")

    def _merge_act_params(self):
        """主进程合并所有rank的act参数到一个文件"""
        merged_act = {}
        for rank in range(self.world_size):
            act_path = os.path.join(self.save_path, f"input_scales_rank{rank}.safetensors")
            if os.path.exists(act_path):
                from safetensors.torch import load_file
                rank_act = load_file(act_path)
                merged_act.update(rank_act)
                os.remove(act_path)  # 合并后删除单个rank的act文件
        
        # 保存合并后的act参数
        if merged_act:
            act_path = os.path.join(self.save_path, "input_scales.safetensors")
            save_file(merged_act, act_path)
            print(f"[Rank 0] 已合并 {len(merged_act)} 个act参数")

    def _cleanup_local_files(self):
        """主进程清理局部映射文件（已合并，无需保留）"""
        for rank in range(self.world_size):
            map_path = os.path.join(self.save_path, f"local_map_rank{rank}.json")
            if os.path.exists(map_path):
                os.remove(map_path)
    
    def copy_files(self):
        names = [
            "generation_config.json", "merges.txt",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
        ]
        # 确保保存目录存在，如果不存在则创建
        os.makedirs(self.save_path, exist_ok=True)
        for name in names:
            # 构建源文件和目标文件的完整路径
            src = os.path.join(self.model_path, name)
            dst = os.path.join(self.save_path, name)
            # 检查源文件是否存在
            if not os.path.exists(src):
                print(f"警告: 源文件 {src} 不存在，跳过复制")
                continue
            # 复制文件
            shutil.copy2(src, dst)

    
    @staticmethod
    def _mapping_act_params(tensor_name):
        replacement_rules = {
            'gate_proj': 'w1',
            'up_proj': 'w3', 
            'down_proj': 'w2'
        }
        if 'experts' not in tensor_name:
            return tensor_name
        for old_pattern, new_pattern in replacement_rules.items():
            if old_pattern in tensor_name:
                return tensor_name.replace(old_pattern, new_pattern)

    def save_quantization_config(self):
        # 1. 读取原始 config.json
        config = self.quant_service.warpped_model.config.copy()
        
        # 2. 添加/更新量化配置
        # del config["quantization_config"]
    
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