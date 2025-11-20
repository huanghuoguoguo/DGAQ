#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 Mamba2 模型用于 DGA 检测
基于官方 mamba_ssm 的 Mamba2 层（SSD算法），无MoE结构
支持二分类和多分类任务

安装依赖:
    pip install mamba-ssm

注意: 需要 CUDA 环境，mamba-ssm 暂不支持 CPU 推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba2
except ImportError:
    raise ImportError(
        "mamba-ssm 未安装或版本过低（需>=1.2.0）。请执行: pip install mamba-ssm\n"
        "注意: 需要 CUDA 环境，暂不支持纯 CPU 运行"
    )


class LightweightMamba2(nn.Module):
    """轻量级 Mamba2 模型用于 DGA 检测
    
    架构:
        - Embedding 层
        - 多层 Mamba2 块 (SSD算法，无注意力机制)
        - 自适应池化 + 分类头
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_length: int = 60,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 d_state: int = 128,           # Mamba2 SSD状态维度（通常比Mamba1大）
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,            # Mamba2新增: 每个头的维度
                 dropout_rate: float = 0.3):
        """
        初始化 Mamba2 模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度（必须能被headdim整除）
            max_length: 最大序列长度（仅用于位置编码）
            num_classes: 分类数量
            num_layers: Mamba2 层数
            d_state: SSD算法状态维度 (默认: 128，是Mamba1的8倍)
            d_conv: 卷积核大小 (默认: 4)
            expand: 扩展系数 (默认: 2)
            headdim: 每个注意力头的维度 (默认: 64)
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 可学习位置编码（Mamba2理论上不需要，但短序列任务中可略微提升）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embedding_dim) * 0.02)
        
        # Mamba2 层堆叠
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            ) for _ in range(num_layers)
        ])
        
        # 层归一化（每层后使用）
        self.norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类头
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 嵌入层 + 位置编码
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]  # [B, L, D]
        x = self.dropout(x)
        
        # Mamba2 层处理
        for mamba2, norm in zip(self.layers, self.norms):
            x = x + self.dropout(norm(mamba2(x)))  # 残差连接
        
        # 平均池化（忽略padding）
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(-1).float()  # [B, L, 1]
        x = (x * padding_mask).sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1)  # [B, D]
        
        # 分类头
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


# ---------------- 自测 ----------------
if __name__ == "__main__":
    print("轻量级 Mamba2 模型")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("⚠️ 警告: mamba-ssm 需要 CUDA 环境，当前为 CPU")
        print("程序将退出")
        exit(1)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 60
    vocab_size = 40
    num_classes = 2
    
    model = LightweightMamba2(
        vocab_size=vocab_size,
        embedding_dim=128,
        max_length=seq_len,
        num_classes=num_classes,
        num_layers=2,
        d_state=128,          # Mamba2 SSD状态维度更大
        expand=2,
        headdim=64
    ).to(device)
    
    # 测试输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试梯度反向传播
    loss = F.cross_entropy(output, torch.randint(0, num_classes, (batch_size,)).to(device))
    loss.backward()
    print(f"损失: {loss.item():.4f}")
    print("✓ 模型测试通过")