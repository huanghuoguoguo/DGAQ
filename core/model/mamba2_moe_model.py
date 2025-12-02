#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级 Mamba2-MoE 模型用于 DGA 检测
架构: Mamba2层 + MoE层 交替堆叠，实现Token级稀疏激活
基于最新的 Mamba2 (SSD算法) + MoE 结构
支持二分类和多分类任务

安装依赖:
    pip install mamba-ssm
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


class MoELayer(nn.Module):
    """改进的序列级MoE层
    
    关键改进：
    1. Top-2路由：融合两个专家的优势
    2. 专家容量优化：减小hidden_dim避免过拟合
    3. 添加专家正则化：鼓励专家差异化
    """
    def __init__(self, d_model: int, num_experts: int = 3, expert_hidden: int = 512, dropout: float = 0.1, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.top_k = min(top_k, num_experts)  # 最多选择top_k个专家

        # 专家网络：更轻量化（避免过拟合）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.GELU(),  # GELU比ReLU更平滑
                nn.Linear(expert_hidden, d_model)
            ) for _ in range(num_experts)
        ])

        # 门控网络：添加层归一化提升稳定性
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_experts)
        )
        
        # 噪声注入（训练时帮助专家探索）
        self.noise_epsilon = 1e-2

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
            gate_weights: [batch_size * seq_len, num_experts]
        """
        B, L, D = x.shape
        x_flat = x.contiguous().view(-1, D)  # [B*L, D]

        # 门控路由（训练时添加噪声）
        gate_logits = self.gate(x_flat)  # [B*L, num_experts]
        
        if self.training:
            # 训练时添加高斯噪声，鼓励探索
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
        
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Top-K路由（默认Top-2）
        top_w, top_idx = torch.topk(gate_weights, k=self.top_k, dim=-1)  # [B*L, top_k]
        
        # 重归一化top-k权重
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        
        # 所有专家并行计算
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*L, num_experts, D]
        
        # 融合top-k个专家的输出
        out_flat = torch.zeros_like(x_flat)  # [B*L, D]
        for k in range(self.top_k):
            expert_idx = top_idx[:, k:k+1]  # [B*L, 1]
            expert_weight = top_w[:, k:k+1]  # [B*L, 1]
            
            # 选择对应专家的输出
            expert_idx_expanded = expert_idx.unsqueeze(-1).expand(-1, -1, D)  # [B*L, 1, D]
            selected_output = torch.gather(expert_outputs, 1, expert_idx_expanded).squeeze(1)  # [B*L, D]
            
            # 加权累加
            out_flat = out_flat + expert_weight * selected_output

        out = out_flat.view(B, L, D)
        return out, gate_weights


class Mamba2MoELayer(nn.Module):
    """单个 Mamba2-MoE 层：Mamba2 + MoE 残差连接"""
    def __init__(self, 
                 d_model: int,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 num_experts: int = 3,
                 expert_hidden: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # Mamba2 核心块 (SSD算法)
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
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
        # Mamba2 分支（Post-Norm，与原版Mamba2保持一致）
        residual = x
        x = self.norm_mamba(self.mamba2(x))
        x = residual + self.dropout(x)
        
        # MoE 分支（Post-Norm）
        residual = x
        moe_out, gate_weights = self.moe(x)
        x = self.norm_moe(moe_out)
        x = residual + self.dropout(x)
        
        return x, gate_weights


class LightweightMamba2MoE(nn.Module):
    """轻量级 Mamba2-MoE 完整模型"""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 256,
                 max_length: int = 60,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 num_experts: int = 3,
                 expert_hidden: int = 512,
                 dropout_rate: float = 0.3,
                 aux_weight: float = 1e-2,     # 辅助损失权重（兼容参数）
                 balance_weight: float = 1e-2):
        super().__init__()
        # 兼容aux_weight参数，实际使用balance_weight
        self.balance_weight = balance_weight if aux_weight == 1e-2 else aux_weight
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts  # 添加此属性供Trainer使用
        
        # 确保embedding_dim是8的倍数（Mamba2 causal_conv1d要求）
        if embedding_dim % 8 != 0:
            raise ValueError(f"embedding_dim必须是8的倍数，当前值: {embedding_dim}")
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embedding_dim) * 0.02)
        
        # 堆叠 Mamba2-MoE 层
        self.layers = nn.ModuleList([
            Mamba2MoELayer(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
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
    
    def forward(self, x: torch.Tensor, return_gate: bool = False):
        # x: [B, L]
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]  # [B, L, D]
        x = self.dropout(x)
        
        # 确保连续性，满足Mamba2 causal_conv1d的stride要求
        x = x.contiguous()
        
        # 逐层处理
        all_gate_weights = []
        for layer in self.layers:
            x = x.contiguous()  # 每层前确保连续
            x, gate_weights = layer(x)
            all_gate_weights.append(gate_weights)
        
        # 平均池化（忽略padding）
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(-1).float()
        x = (x * padding_mask).sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1)
        
        # 分类头
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        if return_gate:
            gate_info = {
                'gate_weights': all_gate_weights,
                'expert_usage': torch.stack([gw.mean(dim=0) for gw in all_gate_weights])
            }
            return x, gate_info
        return x
    
    def compute_loss(self, logits, y_true, gate_info):
        """总损失 = CE + 负载均衡 + 多样性损失
        
        改进的负载均衡策略：
        1. 使用Top-K统计（而非argmax）
        2. 添加专家多样性损失（鼓励专家学习不同模式）
        3. 增强负载均衡惩罚力度
        """
        ce_loss = F.cross_entropy(logits, y_true)
        
        # 负载均衡损失（适配Top-K路由）
        balance_loss = 0.0
        diversity_loss = 0.0
        num_experts = self.layers[0].moe.num_experts
        top_k = self.layers[0].moe.top_k
        
        for gw in gate_info['gate_weights']:
            # gw: [B*L, num_experts] softmax后的门控权重
            
            # === 1. 负载均衡损失（适配Top-K）===
            # 统计每个专家被选入Top-K的频率
            top_idx = torch.topk(gw, k=top_k, dim=-1)[1]  # [B*L, top_k]
            f_i = torch.zeros(num_experts, device=gw.device)
            for i in range(num_experts):
                # 计算专家i出现在top_k中的频率
                f_i[i] = (top_idx == i).any(dim=-1).float().mean()
            
            # 平均门控权重
            p_i = gw.mean(dim=0)  # [num_experts]
            
            # Switch Transformer的负载均衡损失
            # 希望f_i和p_i都接近1/num_experts
            balance_loss += num_experts * (f_i * p_i).sum()
            
            # === 2. 专家多样性损失（新增）===
            # 计算门控权重的方差，方差越大说明专家分化越明显
            # 我们希望方差不要太小（避免所有专家输出相同）
            gate_variance = gw.var(dim=0).mean()
            # 惩罚过小的方差（鼓励专家多样性）
            diversity_loss += torch.relu(0.01 - gate_variance)  # 如果方差<0.01则惩罚
        
        # 平均多层的损失
        balance_loss = balance_loss / len(gate_info['gate_weights'])
        diversity_loss = diversity_loss / len(gate_info['gate_weights'])
        
        # 组合损失（增强负载均衡权重）
        total_loss = ce_loss + self.balance_weight * balance_loss + 0.01 * diversity_loss
        return total_loss


# 自测
if __name__ == "__main__":
    print("轻量级 Mamba2-MoE 模型")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("⚠️ 警告: mamba-ssm 需要 CUDA 环境")
        exit(1)
    
    B, L, V, C = 4, 60, 40, 2
    x = torch.randint(0, V, (B, L)).to(device)
    y = torch.randint(0, C, (B,)).to(device)
    
    model = LightweightMamba2MoE(
        vocab_size=V,
        embedding_dim=256,
        num_classes=C,
        num_layers=2,
        num_experts=3,
        d_state=128,
        headdim=64,
        balance_weight=1e-2
    ).to(device)
    
    model.train()
    logits, info = model(x, return_gate=True)
    loss = model.compute_loss(logits, y, info)
    print(f"训练 - loss: {loss.item():.4f}")
    print(f"专家使用率: {info['expert_usage'].mean(dim=1).tolist()}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"推理 - logits: {logits.shape}")
