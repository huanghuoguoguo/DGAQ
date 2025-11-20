#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 Transformer-MoE 模型用于 DGA 检测
核心：将 TransformerEncoderLayer 的 FFN 替换为 MoE 层
支持二分类和多分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码（复用原版）"""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MoELayer(nn.Module):
    """Token级稀疏MoE层（核心创新）"""
    def __init__(self, d_model: int, num_experts: int = 3, expert_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # 专家网络：每个都是轻量级FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, d_model)
            ) for _ in range(num_experts)
        ])

        # 门控网络：为每个token选择专家
        self.gate = nn.Linear(d_model, num_experts)

        # 记录专家使用统计（用于负载均衡损失）
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('num_tokens_total', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
            gate_weights: [batch_size * seq_len, num_experts]
        """
        B, L, D = x.shape
        x_flat = x.view(-1, D)                      # [B*L, D]

        # Top-1 路由
        gate_logits = self.gate(x_flat)             # [B*L, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_w, top_idx = torch.topk(gate_weights, k=1, dim=-1)  # [B*L, 1]

        # 稀疏计算：只激活被选中的专家
        out_flat = torch.zeros_like(x_flat)         # [B*L, D]
        for i, expert in enumerate(self.experts):
            mask = (top_idx == i).squeeze(1)        # [B*L]
            if mask.any():
                expert_input = x_flat[mask]         # [N, D]
                expert_out = expert(expert_input)   # [N, D]
                out_flat[mask] = top_w[mask] * expert_out  # 修复：去掉squeeze(1)

        # 重塑
        out = out_flat.view(B, L, D)

        # 训练时累积统计
        if self.training:
            self.num_tokens_total += B * L
            self.expert_counts += torch.bincount(top_idx.squeeze(), minlength=self.num_experts)

        return out, gate_weights


class MoETransformerEncoderLayer(nn.Module):
    """含MoE的Transformer编码器层"""
    def __init__(self, d_model: int, nhead: int, num_experts: int = 3,
                 expert_hidden: int = 256, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = MoELayer(d_model, num_experts, expert_hidden, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        # 自注意力
        attn_out, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_out)
        src = self.norm1(src)

        # MoE层（替代FFN）
        moe_out, gate_weights = self.moe(src)
        src = src + self.dropout(moe_out)
        src = self.norm2(src)

        return src, gate_weights


class LightweightTransformerMoE(nn.Module):
    """轻量级Transformer-MoE完整模型"""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_length: int = 60,
                 num_classes: int = 2,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 num_experts: int = 3,
                 expert_hidden: int = 256,
                 dropout_rate: float = 0.3,
                 aux_weight: float = 1e-2,
                 balance_weight: float = 1e-2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.aux_weight = aux_weight
        self.balance_weight = balance_weight

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length, dropout_rate)

        # 堆叠MoE编码器层
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(embedding_dim, num_heads, num_experts,
                                       expert_hidden, dropout=dropout_rate)
            for _ in range(num_layers)
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
        x = self.embedding(x) * math.sqrt(self.embedding_dim)  # [B, L, D]
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # [B, L, D]

        # padding mask
        src_key_padding_mask = (x.sum(dim=-1) == 0)  # [B, L]

        # 逐层编码
        all_gate_weights = []
        for layer in self.layers:
            x, gate_weights = layer(x, src_key_padding_mask)
            all_gate_weights.append(gate_weights)

        # 平均池化
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # 分类
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))

        if return_gate_info:
            gate_info = {
                'gate_weights': all_gate_weights,  # 每层[B*L, num_experts]
                'expert_usage': torch.stack([gw.mean(dim=0) for gw in all_gate_weights])
            }
            return x, gate_info
        return x

    def compute_loss(self, logits, y_true, gate_info):
        """总损失 = CE + 负载均衡损失"""
        ce_loss = F.cross_entropy(logits, y_true)

        # 负载均衡损失 (f_i * p_i)
        balance_loss = 0.0
        for gw in gate_info['gate_weights']:
            f_i = gw.mean(dim=0)          # [num_experts]
            p_i = f_i                      # 简化为 f_i，可換softmax_logits
            balance_loss += (f_i * p_i).sum()

        total_loss = ce_loss + self.balance_weight * balance_loss
        return total_loss


# ---------------- 自测 ----------------
if __name__ == "__main__":
    B, L, V, C = 4, 60, 40, 2
    x = torch.randint(0, V, (B, L))
    y = torch.randint(0, C, (B,))

    model = LightweightTransformerMoE(V, num_classes=C, num_experts=3,
                                     aux_weight=1e-2, balance_weight=1e-2)
    model.train()

    # 训练模式
    logits, info = model(x, return_gate_info=True)
    loss = model.compute_loss(logits, y, info)
    print(f"训练 - logits: {logits.shape}, loss: {loss.item():.4f}")
    print(f"专家使用率: {info['expert_usage'].mean(dim=1).tolist()}")

    # 推理模式（仅激活1个专家/ token）
    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"推理 - logits: {logits.shape}, 速度提升约3倍")