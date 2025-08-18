import torch
import torch.nn as nn


class HookManager:
    """Hook管理器：仅负责注册/移除Hook，不管理校准器（解耦职责）"""
    def __init__(self):
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []  # 存储Hook句柄

    def register_hook(self, layer, quantizer) -> None:
        """为层注册激活值收集Hook，将数据交给校准器"""

        def pre_forward_hook(module, input):
            target_device = module.weight.device
            # 2. 将quantizer移至目标设备（假设quantizer有to方法）
            quantizer.to(target_device)

        # 注册pre_forward_hook（在层forward前完成所有设备同步）
        pre_handle = layer.register_forward_pre_hook(pre_forward_hook)
        self.hook_handles.append(pre_handle)

        def forward_hook(module, input, output):
            # 通常激活值是层的输出（根据层类型调整，这里以output为例）
            input_tensor = input if isinstance(input, torch.Tensor) else input[0]
            weight_tensor = module.weight.data
            quantizer.calibrate(input_tensor, weight_tensor)

        # 注册前向Hook（推荐用forward_hook而非pre_hook，因为激活值通常是输出）
        handle = layer.register_forward_hook(forward_hook)
        self.hook_handles.append(handle)

    def remove_all_hooks(self) -> None:
        """移除所有Hook，避免影响后续计算"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []