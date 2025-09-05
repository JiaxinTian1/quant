import os
import torch
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import shutil


class Qwen3Model:
    def __init__(self, model_path, save_path, dist_ctx):
        self.model_path = model_path
        self.save_path = save_path
        self.dist_ctx = dist_ctx
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            config = self.config,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )

    def forward(self, batch):
        self.model(** batch) 
    
    @staticmethod
    def mapping_tensor_params(tensor_name):
        replacement_rules = {
            'gate_proj': 'w1',
            'up_proj': 'w3', 
            'down_proj': 'w2'
        }
        if 'experts' not in tensor_name:
            return tensor_name
        if 'input_scale' not in tensor_name:
            return tensor_name
        for old_pattern, new_pattern in replacement_rules.items():
            if old_pattern in tensor_name:
                tensor_name = tensor_name.replace(old_pattern, new_pattern)
        return tensor_name
    
    def copy_files(self):
        names = [
            "generation_config.json", "merges.txt",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
        ]
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