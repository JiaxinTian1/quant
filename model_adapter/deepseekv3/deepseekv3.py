import os
import torch
import json
from transformers import AutoTokenizer
from safetensors.torch import load_model

from . import model as deekseep_model
from .kernel import act_quant, weight_dequant, fp8_gemm




class DSV3Model:
    def __init__(self, model_path):
        self.config = None
        self.model = self.load_ds_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=args.trust_remote_code
        )

    def load_ds_model(self, model_path: str):
        """Loads the deepseek model to memory."""
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_config = config_path = os.path.join(current_directory, "config_671B.json")
        # get distributed info
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if world_size > 1:
            dist.init_process_group("nccl")
            torch.cuda.set_device(local_rank)

        # run with bf16
        torch.set_default_dtype(torch.bfloat16)
        # get config and build model
        with open(model_config) as f:
            model_args = deekseep_model.ModelArgs(**json.load(f))
        with open(model_config) as f:
            self.config = json.load(f)
        with torch.device("cuda"):
            model = deekseep_model.Transformer(model_args)

        # load model
        checkpoint_path = os.path.join(model_path, f"model{rank}-mp{world_size}.safetensors")
        print(f"Loading {checkpoint_path}")
        load_model(model, checkpoint_path)
        print(f"Loaded {checkpoint_path}")
        return model

    def forward(self, sbatch):
        self.model.forward(batch)
