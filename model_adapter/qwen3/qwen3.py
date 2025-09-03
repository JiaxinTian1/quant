import torch
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


class Qwen3Model:
    def __init__(self, model_path):
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
        breakpoint()
        self.model(** batch) 