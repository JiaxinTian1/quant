from .sglang_saver import SglangFP8Saver, SglangIN4Saver
from .dist_saver import DistSaver


class ModelSaverFactory:
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