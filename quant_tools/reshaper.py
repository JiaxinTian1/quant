import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class BaseReshaper(ABC):
    """缩放器基类，定义量化参数计算接口"""
    def __init__(self):
        """
        """
        self.original_shape = None


class ReshaperFactory():
    """shaper工厂"""
    @staticmethod
    def create(granularity) -> "BaseScaler":
        if granularity == 'channel':
            return ChannelReshaper()
        elif granularity == 'block':
            return BlockReshaper()
        elif granularity == 'group':
            return GroupReshaper()


class ChannelReshaper(BaseReshaper):
    
    def reshape(self, tensor: torch.Tensor):
        return tensor
    
    def unreshape(self, tensor: torch.Tensor):
        return tensor

class BlockReshaper(BaseReshaper):
    
    def reshape(self, tensor: torch.Tensor, block_size: tuple = (128, 128)):
        """
        计算矩阵的 Per-Block 最大值 (amax) 矩阵
        
        Args:
            tensor: 输入矩阵 (形状 [H, W]，如 4096x4096)
            block_size: 分块大小 (height, width)，如 (128, 128)
        
        Returns:
            amax_matrix: 每个块的最大值矩阵 (形状 [H//block_H, W//block_W]，如 32x32)
        """
        # 检查输入合法性
        assert len(tensor.shape) == 2, "输入必须是2D矩阵"
        H, W = tensor.shape
        block_H, block_W = block_size
        assert H % block_H == 0 and W % block_W == 0, "矩阵尺寸必须能被分块大小整除"

        self.original_shape = tensor.shape
        
        # 1. 将矩阵划分为块 -> (num_blocks_H, block_H, num_blocks_W, block_W)
        reshaped = tensor.view(H // block_H, block_H, W // block_W, block_W)
        
        # 2. 交换维度以分离空间和块内维度 -> (num_blocks_H, num_blocks_W, block_H, block_W)
        reshaped = reshaped.permute(0, 2, 1, 3)
        reshaped = reshaped.reshape(H // block_H, W // block_W, -1)
        return reshaped
    
    def unreshape(self, tensor: torch.Tensor, block_size: tuple = (128, 128)):
        """
        将分块后的张量恢复为原始形状
        
        Args:
            reshaped: 分块后的张量 [num_blocks_H, num_blocks_W, block_H * block_W]
            original_shape: 原始矩阵形状 (H, W)
            block_size: 分块大小 (block_H, block_W)
        """
        H, W = self.original_shape
        block_H, block_W = block_size
        num_blocks_H, num_blocks_W = H // block_H, W // block_W
        
        # 1. 恢复块内维度 -> [num_blocks_H, num_blocks_W, block_H, block_W]
        unreshaped = tensor.view(num_blocks_H, num_blocks_W, block_H, block_W)
        # 2. 交换维度 -> [num_blocks_H, block_H, num_blocks_W, block_W]
        unreshaped = unreshaped.permute(0, 2, 1, 3)
        # 3. 合并为原始形状 -> [H, W]
        unreshaped = unreshaped.reshape(H, W)
        return unreshaped


class GroupReshaper(BaseReshaper):
    
    def reshape(self, tensor: torch.Tensor, group_size: int = 128):
        """
        按最后一维进行分组reshape
        
        Args:
            tensor: 输入张量 (形状 [..., D]，如 [2, 7, 4096])
            group_size: 分组大小，如128
        
        Returns:
            reshaped: 分组后的张量 (形状 [..., D//group_size, group_size]，如 [2, 7, 32, 128])
        """
        # 检查输入合法性
        assert tensor.dim() >= 1, "输入至少要有1维"
        D = tensor.shape[-1]
        assert D % group_size == 0, f"最后一维大小{D}必须能被group_size{group_size}整除"

        self.original_shape = tensor.shape
        self.group_size = group_size
        
        # 计算新的形状
        new_shape = list(tensor.shape[:-1]) + [D // group_size, group_size]
        reshaped = tensor.reshape(new_shape)
        return reshaped
    
    def unreshape(self, tensor: torch.Tensor):
        """
        将分组后的张量恢复为原始形状
        
        Args:
            tensor: 分组后的张量 [..., groups, group_size]
        
        Returns:
            unreshaped: 原始形状的张量 [..., D]
        """
        # 恢复原始形状
        unreshaped = tensor.reshape(self.original_shape)
        return unreshaped