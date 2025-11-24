#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析MoE模型的专家使用情况
检测专家塌陷问题
"""

import torch
import torch.nn as nn
from core.model.mamba2_moe_model import LightweightMamba2MoE
from core.data_loader import DGADataset
from torch.utils.data import DataLoader

def analyze_expert_usage(model, data_loader, device, num_batches=100):
    """分析专家使用情况"""
    model.eval()
    
    num_experts = model.num_experts
    num_layers = len(model.layers)
    
    # 统计每层每个专家的使用次数
    expert_counts = torch.zeros(num_layers, num_experts).to(device)
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            data = data.to(device)
            outputs, gate_info = model(data, return_gate=True)
            
            # 统计每层的专家选择
            for layer_idx, gate_weights in enumerate(gate_info['gate_weights']):
                # gate_weights: [B*L, num_experts]
                top_idx = torch.argmax(gate_weights, dim=-1)  # [B*L]
                for expert_idx in range(num_experts):
                    expert_counts[layer_idx, expert_idx] += (top_idx == expert_idx).sum()
                
                total_tokens += gate_weights.size(0)
    
    # 计算百分比
    expert_percentages = expert_counts / (total_tokens / num_layers) * 100
    
    return expert_percentages.cpu().numpy()

def print_expert_analysis(expert_percentages):
    """打印专家使用分析"""
    num_layers, num_experts = expert_percentages.shape
    expected = 100.0 / num_experts
    
    print("\n" + "="*70)
    print("📊 MoE 专家使用情况分析")
    print("="*70)
    
    for layer_idx in range(num_layers):
        print(f"\n第 {layer_idx+1} 层:")
        print(f"  期望使用率: {expected:.2f}% (均衡状态)")
        print(f"  实际使用率:")
        
        for expert_idx in range(num_experts):
            usage = expert_percentages[layer_idx, expert_idx]
            deviation = abs(usage - expected)
            
            # 可视化条形图
            bar_len = int(usage / 2)  # 除以2是为了适应50字符宽度
            bar = '█' * bar_len + '░' * (50 - bar_len)
            
            # 颜色标记（用emoji表示）
            if usage < 5:
                status = "🔴 塌陷"
            elif deviation > 20:
                status = "🟡 不均衡"
            elif deviation > 10:
                status = "🟢 偏差"
            else:
                status = "✅ 正常"
            
            print(f"    专家 {expert_idx+1}: [{bar}] {usage:5.2f}% {status}")
    
    # 整体分析
    print("\n" + "-"*70)
    print("整体分析:")
    
    # 检测塌陷
    collapsed = (expert_percentages < 5).any(axis=0)
    if collapsed.any():
        collapsed_experts = [i+1 for i, c in enumerate(collapsed) if c]
        print(f"  ⚠️  专家塌陷: 专家 {collapsed_experts} 几乎不被使用 (<5%)")
    else:
        print(f"  ✅ 无专家塌陷")
    
    # 检测负载不均衡
    max_imbalance = abs(expert_percentages - expected).max()
    print(f"  最大负载偏差: {max_imbalance:.2f}%")
    
    if max_imbalance > 30:
        print(f"  ⚠️  严重不均衡! 建议降低 balance_weight")
    elif max_imbalance > 20:
        print(f"  🟡 轻微不均衡，可以优化")
    else:
        print(f"  ✅ 负载均衡良好")
    
    # 计算方差
    variance = expert_percentages.var(axis=1).mean()
    print(f"  平均方差: {variance:.2f}")
    
    print("="*70)

if __name__ == "__main__":
    # 加载数据
    print("加载数据集...")
    dataset = DGADataset()
    dataset.load('/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl')
    
    _, val_loader, _ = dataset.get_loaders(batch_size=512, task_type='binary')
    
    # 创建模型
    print("创建Mamba2-MoE模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LightweightMamba2MoE(
        vocab_size=dataset.vocab_size,
        embedding_dim=256,
        max_length=dataset.max_length,
        num_classes=2,
        num_layers=2,
        d_state=128,
        headdim=64,
        num_experts=4,
        expert_hidden=256,
        dropout_rate=0.15,
        balance_weight=0.001
    ).to(device)
    
    # 尝试加载训练好的模型
    try:
        model.load_state_dict(torch.load('./models/mamba2_moe_binary_model.pth'))
        print("✅ 加载训练好的模型")
    except:
        print("⚠️  未找到训练好的模型，使用随机初始化")
    
    # 分析专家使用情况
    print("\n分析专家使用情况（验证集前100批次）...")
    expert_percentages = analyze_expert_usage(model, val_loader, device, num_batches=100)
    
    # 打印分析结果
    print_expert_analysis(expert_percentages)
