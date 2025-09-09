import torch
import torch.nn as nn
import torch.distributed as dist
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
        self.local_map = {}
        
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
            self.warpped_model.save_quantization_config()
            self.warpped_model.save_hf_quant_config()
            self.warpped_model.copy_files()
            self._cleanup_local_files()  # 清理局部映射文件
            print(f"[Rank 0] 保存完成，目录: {self.save_path}")

    def _save_rank_files(self):
        """当前进程直接保存分片和局部映射到最终目录"""
        model_state_dict = self._process_state_dict(self.warpped_model.model.state_dict())
        
        self._collect_act_params(model_state_dict)
        self._collect_weight_params(model_state_dict)
        
        self._save_local_map()

    def _process_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """收集当前进程的state_dict"""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = self.warpped_model.mapping_tensor_params(key)
            processed_tensor = self.warpped_model.merge_tp_shards_with_gather(new_key, value)
            if processed_tensor is not None and processed_tensor.numel() > 0:
                new_state_dict[new_key] = processed_tensor
        return new_state_dict
    
    def _collect_act_params(self, state_dict: Dict[str, torch.Tensor]):
        """收集当前进程的act参数"""
        act_params = {}
        act_filename = f"input_scales_rank{self.rank}.safetensors"
        for key, value in state_dict.items():
            if "input_scale" in key:
                act_params[key] = value
                self.local_map[key] = "input_scales.safetensors"
        if act_params:
            act_path = os.path.join(self.save_path, act_filename)
            save_file(act_params, act_path)
    
    def _collect_weight_params(self, state_dict: Dict[str, torch.Tensor]):
        chunk_params = {}
        chunk_id = self.rank + 1
        total_chunks = self.world_size
        chunk_filename = f"model-{chunk_id:05d}-of-{total_chunks:05d}.safetensors"
        for key, value in state_dict.items():
            if "input_scale" not in key:
                chunk_params[key] = value
                self.local_map[key] = chunk_filename
        if chunk_params:
            chunk_path = os.path.join(self.save_path, chunk_filename)
            save_file(chunk_params, chunk_path)

    def _save_local_map(self):
        """保存当前进程的局部权重映射（供主进程合并）"""
        # 保存局部映射（文件名含rank，避免冲突）
        map_path = os.path.join(self.save_path, f"local_map_rank{self.rank}.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.local_map, f, indent=4)

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