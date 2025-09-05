import os
import torch
import torch.distributed as dist
import json
import shutil
from transformers import AutoTokenizer
from safetensors.torch import load_model

from . import model as deekseep_model
from .kernel import act_quant, weight_dequant, fp8_gemm




class DSV3Model:
    def __init__(self, model_path, save_path, dist_ctx):
        self.model_path = model_path
        self.save_path = save_path
        self.dist_ctx = dist_ctx
        self.config = None
        self.model = self.load_ds_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def load_ds_model(self):
        """Loads the deepseek model to memory."""
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_config = config_path = os.path.join(current_directory, "config_671B.json")
        with open(model_config) as f:
            self.config = json.load(f)


        # run with bf16
        torch.set_default_dtype(torch.bfloat16)
        # get config and build model
        model_args = deekseep_model.ModelArgs(** self.config)
        
        with torch.device(self.dist_ctx.device):
            model = deekseep_model.Transformer(model_args)

        # load model
        checkpoint_name = f"model{self.dist_ctx.rank}-mp{self.dist_ctx.world_size}.safetensors"
        checkpoint_path = os.path.join(self.model_path, checkpoint_name)
        print(f"Loading {checkpoint_path}")
        load_model(model, checkpoint_path)
        print(f"Loaded {checkpoint_path}")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, batch):
        # breakpoint()
        self.model(batch["input_ids"])
    
    @staticmethod
    def mapping_tensor_params(tensor_name):
        replacement_rules = {
            'embed': 'embed_tokens',
            'attn_norm': 'input_layernorm',
            'ffn_norm': 'post_attention_layernorm',
            'wq_a': 'q_a_proj',
            'q_norm': 'q_a_layernorm',
            'wq_b': 'q_b_proj',
            'wkv_a': 'kv_a_proj_with_mqa',
            'kv_norm': 'kv_a_layernorm',
            'wkv_b': 'kv_b_proj',
            'wo': 'o_proj',
            'head': 'lm_head',
            'gate.bias': 'gate.e_score_correction_bias',
            '.attn.': '.self_attn.',
            '.ffn.': '.mlp.',

        }
        weight_replacement_rules = {
            'w1': 'gate_proj',
            'w2': 'down_proj',
            'w3': 'up_proj',
            '.scale': '.weight_scale_inv',
        }
        if 'input_scale' not in tensor_name:
            replacement_rules.update(weight_replacement_rules)
        for old_pattern, new_pattern in replacement_rules.items():
            if old_pattern in tensor_name:
                tensor_name = tensor_name.replace(old_pattern, new_pattern)
        if 'lm_head' not in tensor_name:
            tensor_name = "model." + tensor_name
        return tensor_name
    
    def merge_tp_shards_with_gather(self, key, tensor):
        if 'shared_experts' not in key:
            return tensor
        world_size = self.dist_ctx.world_size
        rank = self.dist_ctx.rank
        gathered_tensor = []
        if rank == 0:
            shard_shape = list(tensor.shape)  # 单个分片的形状（如 7168x256）
            # 为每个进程的分片预分配内存（共 8 个）
            for _ in range(world_size):
                # 注意：这里应使用分片形状，而非完整形状（完整形状是拼接后的）
                shard = torch.empty(shard_shape, device=tensor.device, dtype=tensor.dtype)
                gathered_tensor.append(shard)
        dist.gather(
            tensor,                # 当前 rank 的分片
            gather_list=gathered_tensor if rank == 0 else None,  # 输出缓冲区（仅 rank0）
            dst=0                  # 目标 rank
        )
        if rank == 0:
            dim = 1 if 'down_proj' in key else 0
            merged_tensor = torch.cat(gathered_tensor, dim=dim)
            return merged_tensor
        else:
            return None

    def copy_files(self):
        names = [
            "generation_config.json", "merges.txt",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "configuration_deepseek.py", "modeling_deepseek.py"
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


    def save_quantization_config(self):
        original_config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(original_config_path):
            print(f"警告: 源文件 {original_config_path} 不存在，跳过复制")
            return
        
        with open(original_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "quantization_config" in config:
            del config["quantization_config"]
        
        target_config_path = os.path.join(self.save_path, "config.json")
        with open(target_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)


    def save_hf_quant_config(self):
        """保存量化参数（保持原有逻辑，无需修改）"""
        hf_quant_config = {}
        hf_quant_config['quantization'] = {}
        hf_quant_config['quantization']["quant_algo"] = "MIXED_PRECISION"
        hf_quant_config['quantization']["kv_cache_quant_algo"] = None
        with open(os.path.join(self.save_path, "hf_quant_config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_quant_config, f, indent=4, ensure_ascii=False)
