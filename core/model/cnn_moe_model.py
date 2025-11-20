#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 CNN-MoE 模型用于 DGA 检测
二分类 / 多分类皆可；Top-1 稀疏路由 + Load-Balance 辅助损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any



class ExpertCNN(nn.Module):
    """单专家：结构与 LightweightCNN 主干相同，但宽度缩小"""
    def __init__(self, embedding_dim: int = 128, channels: tuple = (64, 32, 16)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv1d(embedding_dim, c1, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(c3)
        self.pool  = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, emb, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.pool(x).squeeze(-1)          # (B, c3)


class GatingNetwork(nn.Module):
    """Top-1 门控：输入同样嵌入 -> 选专家"""
    def __init__(self, embedding_dim: int = 128, num_experts: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_experts = num_experts
        self.feat = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_experts)
        )

    def forward(self, x: torch.Tensor, training: bool = True):
        # x: (B, emb, L)
        h = self.feat(x).squeeze(-1)            # (B, 64)
        logits = self.gate(h)                   # (B, num_experts)

        if training:                            # Gumbel 噪声鼓励探索
            noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + 0.1 * noise
        gates = F.softmax(logits, dim=-1)

        top_w, top_idx = torch.topk(gates, k=1, dim=-1)   # (B, 1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)   # 归一化
        return top_w.squeeze(-1), top_idx.squeeze(-1), gates


class LightweightCNNMoE(nn.Module):
    """轻量 CNN-MoE：接口与 LightweightCNN 完全一致"""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_length: int = 60,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 num_experts: int = 3,
                 aux_weight: float = 1e-2,
                 balance_weight: float = 1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.aux_weight = aux_weight
        self.balance_weight = balance_weight

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gating = GatingNetwork(embedding_dim, num_experts, dropout_rate)

        # 专家网络（宽度递减，保证总参数量 ≈ 原单模型）
        self.experts = nn.ModuleList([
            ExpertCNN(embedding_dim, channels=(64, 32, 16)) for _ in range(num_experts)
        ])

        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, dga_type_label: torch.Tensor = None, return_gate: bool = False):
        # x: (B, L)
        emb = self.embedding(x).transpose(1, 2)          # (B, emb, L)

        top_w, top_idx, gates = self.gating(emb, training=self.training)
        batch_size = emb.size(0)

        # 稀疏专家计算
        expert_out = torch.zeros(batch_size, 16, device=emb.device)
        if self.training:
            for i, expert in enumerate(self.experts):
                out = expert(emb)                              # (B, 16)
                mask = (top_idx == i).float().unsqueeze(-1)
                expert_out += mask * out
        else:
            for i, expert in enumerate(self.experts):
                mask = (top_idx == i)
                if mask.any():
                    expert_out[mask] = expert(emb[mask])

        # 分类
        out = self.dropout(expert_out)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)

        if return_gate:
            return logits, {'gates': gates, 'top_idx': top_idx,
                           'expert_usage': gates.mean(dim=0)}
        return logits

    def compute_loss(self, logits, y_true, gate_info):
        ce_loss = F.cross_entropy(logits, y_true)

        # Load-Balance 损失
        f_i = gate_info['gates'].mean(dim=0)
        balance_loss = (f_i * gate_info['gates'].mean(dim=0)).sum()

        # 可选：DGA 类型辅助损失（若传入硬标签）
        aux_loss = 0.0
        if 'dga_type_label' in gate_info:
            aux_loss = F.cross_entropy(gate_info['gates'], gate_info['dga_type_label'])

        total = ce_loss + self.balance_weight * balance_loss + self.aux_weight * aux_loss
        return total




# ---------------- 快速自检 ----------------
if __name__ == "__main__":
    B, L, V = 4, 60, 128
    x = torch.randint(0, V, (B, L))
    y = torch.randint(0, 2, (B,))
    dga_type = torch.randint(0, 2, (B,))          # 0=字典型 1=字符型

    model = LightweightCNNMoE(V, num_classes=2, num_experts=2,
                             aux_weight=0.3, balance_weight=1e-2)
    model.train()
    logits, info = model(x, dga_type_label=dga_type, return_gate=True)
    loss = model.compute_loss(logits, y, info)
    print('loss:', loss.item())
    print('expert_usage:', info['expert_usage'].tolist())

    model.eval()
    with torch.no_grad():
        logits = model(x)
        print('eval_logits_shape:', logits.shape)