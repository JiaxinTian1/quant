# --------------------------
# 量化层实现
# --------------------------
class QuantLinear(nn.Module):
    """继承nn.Linear的量化Linear层，复用基础功能"""
    def __init__(self):
        super().__init__() 
        # 量化相关参数
        self.is_training = False
        self.weight_quant = None
        
    
    def set_params(self, quantizer):
        self.algo = quantizer.algo
        self.in_features = quantizer.layer.in_features
        self.out_features = quantizer.layer.out_features
        self.act_params = quantizer.quant_info["activation"]
        self.weight_params = quantizer.quant_info["weight"]
        self.weight = nn.Parameter(quantizer.layer.weight.data)
        try:
            self.bias = nn.Parameter(quantizer.layer.bias.data)
        except AttributeError:
            self.bias = None
        finally:
            pass
    
    def quantize_weight(self):
        self.weight_quant = self.quantizer.quantize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_training:
            return self._forward_pseudo_quant(x)
        else:
            return self._forward_true_quant(x)

    def _forward_pseudo_quant(self, x: torch.Tensor) -> torch.Tensor:
        """伪量化逻辑：复用父类的bias，量化仅影响weight和activation"""
        # 激活伪量化
        x_quant = self.algo.act_calibrator.quantize_weight(x, self.act_params)
        x_dequant = self.algo.act_calibrator.dequantize_weight(x_quant, self.act_params)

        # 权重伪量化（反量化）
        weight_scale = self.weight_params.get("scale", 1.0)
        weight_zp = self.weight_params.get("offset", 0)
        weight_dequant = (self.weight_quant - weight_zp) * weight_scale

        # 调用父类的线性计算逻辑（复用bias）
        y = torch.matmul(x_dequant, weight_dequant.t())
        if self.bias is not None:
            y += self.bias  # 直接使用父类的bias
        
        return y

    def _forward_true_quant(self, x: torch.Tensor) -> torch.Tensor:
        """真量化逻辑：低精度计算"""
        # 激活量化
        act_scale = self.act_params.get("scale", 1.0)
        act_zp = self.act_params.get("offset", 0)
        x_quant = torch.clamp(
            torch.round(x / act_scale) + act_zp,
            min=self.act_params.get("dtype", torch.int8).min,
            max=self.act_params.get("dtype", torch.int8).max
        ).to(self.act_params.get("dtype", torch.int8))

        # 权重直接使用量化值
        weight_quant = self.weight_quant

        # 低精度矩阵乘
        y_int = torch.matmul(
            x_quant.to(torch.int32),
            weight_quant.to(torch.int32).t()
        )

        # 应用缩放和bias（复用父类的bias）
        joint_scale = act_scale * self.weight_params.get("scale", 1.0)
        y = y_int.to(torch.float32) * joint_scale
        if self.bias is not None:
            y += self.bias
        
        return y