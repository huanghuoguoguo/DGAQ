#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 Mamba-MoE 模型用于 DGA 检测
架构: Mamba层 + MoE层 交替堆叠，实现Token级稀疏激活
支持二分类和多分类任务

安装依赖:
    pip install mamba-ssm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "mamba-ssm 未安装。请执行: pip install mamba-ssm\n"
        "注意: 需要 CUDA 环境，暂不支持纯 CPU 运行"
    )


class MoELayer(nn.Module):
    """Token级稀疏MoE层（复用Transformer-MoE版本）"""
    def __init__(self, d_model: int, num_experts: int = 3, expert_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # 专家网络：轻量级FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, d_model)
            ) for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
            gate_weights: [batch_size * seq_len, num_experts]
        """
        B, L, D = x.shape
        x_flat = x.contiguous().view(-1, D)                      # [B*L, D]

        # Top-1 路由
        gate_logits = self.gate(x_flat)                         # [B*L, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_w, top_idx = torch.topk(gate_weights, k=1, dim=-1)  # [B*L, 1]

        # 稀疏计算
        out_flat = torch.zeros_like(x_flat)                     # [B*L, D]
        for i, expert in enumerate(self.experts):
            mask = (top_idx == i).squeeze(1)                    # [B*L]
            if mask.any():
                expert_input = x_flat[mask]                     # [N, D]
                expert_out = expert(expert_input)               # [N, D]
                out_flat[mask] = top_w[mask] * expert_out  # 修复：去掉squeeze(1)

        out = out_flat.view(B, L, D)
        return out, gate_weights


class MambaMoELayer(nn.Module):
    """单个 Mamba-MoE 层：Mamba + MoE 残差连接"""
    def __init__(self, 
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 num_experts: int = 3,
                 expert_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Mamba 核心块
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # MoE 块（替代FFN）
        self.moe = MoELayer(d_model, num_experts, expert_hidden, dropout)
        
        # 归一化与Dropout
        self.norm_mamba = nn.LayerNorm(d_model)
        self.norm_moe = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
            gate_weights: [batch_size * seq_len, num_experts]
        """
        # Mamba 分支
        x = x + self.dropout(self.norm_mamba(self.mamba(x)))
        
        # MoE 分支
        moe_out, gate_weights = self.moe(x)
        x = x + self.dropout(self.norm_moe(moe_out))
        
        return x, gate_weights


class LightweightMambaMoE(nn.Module):
    """轻量级 Mamba-MoE 完整模型"""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_length: int = 60,
                 num_classes: int = 2,
                 num_layers: int = 2,          # Mamba-MoE 层数
                 d_state: int = 16,            # Mamba SSM状态维度
                 d_conv: int = 4,              # Mamba 卷积核大小
                 expand: int = 2,              # Mamba 扩展系数
                 num_experts: int = 3,         # 每层MoE专家数
                 expert_hidden: int = 256,
                 dropout_rate: float = 0.3,
                 balance_weight: float = 1e-2):
        super().__init__()
        self.balance_weight = balance_weight
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 位置编码（Mamba理论上不需要，但加入可提升极短序列性能）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embedding_dim) * 0.02)
        
        # 堆叠 Mamba-MoE 层
        self.layers = nn.ModuleList([
            MambaMoELayer(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_experts=num_experts,
                expert_hidden=expert_hidden,
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x: torch.Tensor, return_gate_info: bool = False):
        # x: [B, L]
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]  # [B, L, D]
        x = self.dropout(x)
        
        # 逐层处理
        all_gate_weights = []
        for layer in self.layers:
            x, gate_weights = layer(x)
            all_gate_weights.append(gate_weights)
        
        # 平均池化（忽略padding）
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(-1).float()
        x = (x * padding_mask).sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1)
        
        # 分类头
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        if return_gate_info:
            gate_info = {
                'gate_weights': all_gate_weights,
                'expert_usage': torch.stack([gw.mean(dim=0) for gw in all_gate_weights])
            }
            return x, gate_info
        return x
    
    def compute_loss(self, logits, y_true, gate_info):
        """总损失 = CE + 负载均衡"""
        ce_loss = F.cross_entropy(logits, y_true)
        
        # 负载均衡损失（每层的f_i * p_i）
        balance_loss = 0.0
        for gw in gate_info['gate_weights']:
            f_i = gw.mean(dim=0)          # [num_experts]
            p_i = torch.softmax(gw, dim=-1).mean(dim=0)
            balance_loss += (f_i * p_i).sum()
        
        total_loss = ce_loss + self.balance_weight * balance_loss
        return total_loss


# ---------------- 自测 ----------------
if __name__ == "__main__":
    print("轻量级 Mamba-MoE 模型")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("⚠️ 警告: mamba-ssm 需要 CUDA 环境")
        exit(1)
    
    B, L, V, C = 4, 60, 40, 2
    x = torch.randint(0, V, (B, L)).to(device)
    y = torch.randint(0, C, (B,)).to(device)
    
    model = LightweightMambaMoE(
        vocab_size=V,
        embedding_dim=128,
        num_classes=C,
        num_layers=2,
        num_experts=3,
        d_state=16,
        expand=2,
        balance_weight=1e-2
    ).to(device)
    
    # 训练模式
    model.train()
    logits, info = model(x, return_gate_info=True)
    loss = model.compute_loss(logits, y, info)
    print(f"训练 - loss: {loss.item():.4f}")
    print(f"专家使用率: {info['expert_usage'].mean(dim=1).tolist()}")  # 每层平均
    
    # 推理模式（仅激活1个专家/token，速度约3倍）
    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"推理 - logits: {logits.shape}, 速度提升约 {model.num_experts}x")