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

from .savers.sglang_saver import SglangFP8Saver, SglangIN4Saver
from .savers.dist_saver import DistSaver

class SaverFactory:
    """保存器工厂"""
    @staticmethod
    def create(quant_service):
        if quant_service.quant_target == "sglang":
            if quant_service.dist_ctx.world_size > 1:
                return DistSaver(quant_service)
            if quant_service.quant_type == "fp8_e4m3":
                return SglangFP8Saver(quant_service)
            elif quant_service.quant_type == "int4":
                return SglangIN4Saver(quant_service)


