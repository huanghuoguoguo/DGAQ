#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Transformer 的轻量级模型用于 DGA 检测
支持二分类和多分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LightweightTransformer(nn.Module):
    """轻量级 Transformer 模型用于 DGA 检测"""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_length: int = 60,
                 num_classes: int = 2,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout_rate: float = 0.3):
        """
        初始化 Transformer 模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度（必须能被 num_heads 整除）
            max_length: 最大序列长度
            num_classes: 分类数量
            num_heads: 多头注意力的头数
            num_layers: Transformer 层数
            dim_feedforward: 前馈网络维度
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length, dropout_rate)
        
        # Transformer 编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True  # 使用 batch_first=True 简化处理
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 初始化权重
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
        # 嵌入层 [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        
        # 缩放嵌入（标准 Transformer 做法）
        x = x * math.sqrt(self.embedding_dim)
        
        # 位置编码需要 [seq_len, batch_size, embedding_dim]
        # 由于我们使用 batch_first=True，需要转置
        x = x.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]
        
        # 创建 padding mask (0 表示 padding 位置)
        src_key_padding_mask = (x.sum(dim=-1) == 0)  # [batch_size, seq_len]
        
        # Transformer 编码
        # batch_first=True 时输入输出都是 [batch_size, seq_len, embedding_dim]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 使用 [CLS] token 的表示（取第一个位置）或者平均池化
        # 这里使用平均池化（考虑 mask）
        mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch_size, embedding_dim]
        
        # 分类头
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    # 测试模型
    print("轻量级 Transformer 模型")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 60
    vocab_size = 40
    num_classes = 2
    
    model = LightweightTransformer(
        vocab_size=vocab_size,
        embedding_dim=128,
        max_length=seq_len,
        num_classes=num_classes,
        num_heads=4,
        num_layers=2
    )
    
    # 测试输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试梯度反向传播
    loss = F.cross_entropy(output, torch.randint(0, num_classes, (batch_size,)))
    loss.backward()
    print(f"损失: {loss.item():.4f}")
    print("✓ 模型测试通过")
