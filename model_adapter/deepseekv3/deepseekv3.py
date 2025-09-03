import os
import torch
import torch.distributed as dist
import json
from transformers import AutoTokenizer
from safetensors.torch import load_model

from . import model as deekseep_model
from .kernel import act_quant, weight_dequant, fp8_gemm




class DSV3Model:
    def __init__(self, model_path, dist_ctx):
        self.model_path = model_path
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
