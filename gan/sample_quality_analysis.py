#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAN生成样本质量分析
评估生成的对抗样本与真实DGA样本的相似度
"""

import os
import sys
import torch
import numpy as np
from collections import Counter
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.adversarial.generator import DGAGenerator
from core.dataset import load_dataset
from core.model.cnn_model import LightweightCNN

def load_generator(model_path, device, vocab_size=41, hidden_dim=256, max_len=60, z_dim=100):
    """加载生成器"""
    generator = DGAGenerator(vocab_size=vocab_size, hidden_dim=hidden_dim, 
                            max_len=max_len, z_dim=z_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    generator.eval()
    return generator

def analyze_sequence_distribution(sequences):
    """分析序列统计特征"""
    sequences_np = sequences.cpu().numpy() if torch.is_tensor(sequences) else sequences
    
    # 有效长度（非padding部分）
    lengths = []
    for seq in sequences_np:
        length = np.sum(seq != 0)
        lengths.append(length)
    
    # 字符分布
    all_chars = sequences_np[sequences_np != 0]
    char_dist = Counter(all_chars)
    
    # 唯一字符数
    unique_chars = len(char_dist)
    
    return {
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'unique_chars': unique_chars,
        'char_distribution': dict(sorted(char_dist.items())[:10]),  # Top 10
        'total_chars': len(all_chars)
    }

def main():
    parser = argparse.ArgumentParser(description="分析GAN生成样本质量")
    parser.add_argument('--generator_path', type=str, 
                       default='./models/gan/generator_epoch_10.pth')
    parser.add_argument('--dataset_path', type=str,
                       default='./data/processed/500k_unified_dga_dataset.pkl')
    parser.add_argument('--target_model_path', type=str,
                       default='./models/cnn_binary_model.pth')
    parser.add_argument('--num_samples', type=int, default=5000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载数据集
    print("加载真实数据集...")
    dataset = load_dataset(args.dataset_path)
    
    # 获取真实DGA样本（标签>0的为恶意）
    from core.dataset import create_data_loaders
    _, _, test_loader, info = create_data_loaders(args.dataset_path, 
                                                   batch_size=args.num_samples,
                                                   task_type='binary')
    
    # 获取一批测试数据
    for real_seqs, labels in test_loader:
        # 只取恶意样本
        malicious_mask = labels == 1
        real_dga_samples = real_seqs[malicious_mask][:args.num_samples]
        benign_samples = real_seqs[labels == 0][:args.num_samples]
        break
    
    print(f"真实DGA样本数: {len(real_dga_samples)}")
    print(f"真实良性样本数: {len(benign_samples)}\n")
    
    # 加载生成器
    print(f"加载生成器: {args.generator_path}")
    generator = load_generator(args.generator_path, device)
    
    # 生成对抗样本
    print(f"生成 {args.num_samples} 个对抗样本...")
    with torch.no_grad():
        generated_samples = generator.sample(args.num_samples, device)
    
    # 加载目标分类器
    print(f"加载目标分类器: {args.target_model_path}\n")
    target_model = LightweightCNN(vocab_size=41, embedding_dim=128, 
                                   max_length=60, num_classes=2).to(device)
    target_model.load_state_dict(torch.load(args.target_model_path, 
                                             map_location=device, weights_only=False))
    target_model.eval()
    
    # 分析统计特征
    print("="*70)
    print("📊 样本统计特征对比")
    print("="*70)
    
    gen_stats = analyze_sequence_distribution(generated_samples)
    real_dga_stats = analyze_sequence_distribution(real_dga_samples)
    benign_stats = analyze_sequence_distribution(benign_samples)
    
    print(f"\n{'指标':<20} {'生成样本':>15} {'真实DGA':>15} {'真实良性':>15}")
    print("-"*70)
    print(f"{'平均长度':<20} {gen_stats['avg_length']:>15.2f} {real_dga_stats['avg_length']:>15.2f} {benign_stats['avg_length']:>15.2f}")
    print(f"{'长度标准差':<20} {gen_stats['std_length']:>15.2f} {real_dga_stats['std_length']:>15.2f} {benign_stats['std_length']:>15.2f}")
    print(f"{'最小长度':<20} {gen_stats['min_length']:>15.0f} {real_dga_stats['min_length']:>15.0f} {benign_stats['min_length']:>15.0f}")
    print(f"{'最大长度':<20} {gen_stats['max_length']:>15.0f} {real_dga_stats['max_length']:>15.0f} {benign_stats['max_length']:>15.0f}")
    print(f"{'唯一字符数':<20} {gen_stats['unique_chars']:>15} {real_dga_stats['unique_chars']:>15} {benign_stats['unique_chars']:>15}")
    
    # 分类器预测分布
    print(f"\n{'='*70}")
    print("🎯 分类器预测分析")
    print("="*70)
    
    with torch.no_grad():
        gen_logits = target_model(generated_samples)
        real_dga_logits = target_model(real_dga_samples.to(device))
        benign_logits = target_model(benign_samples.to(device))
        
        gen_probs = torch.softmax(gen_logits, dim=1)
        real_dga_probs = torch.softmax(real_dga_logits, dim=1)
        benign_probs = torch.softmax(benign_logits, dim=1)
        
        gen_preds = torch.argmax(gen_probs, dim=1)
        real_dga_preds = torch.argmax(real_dga_probs, dim=1)
        benign_preds = torch.argmax(benign_probs, dim=1)
    
    print(f"\n{'样本类型':<20} {'预测为良性':>15} {'预测为恶意':>15} {'准确率/ASR':>15}")
    print("-"*70)
    
    gen_benign = (gen_preds == 0).sum().item()
    gen_malicious = (gen_preds == 1).sum().item()
    asr = gen_benign / len(generated_samples) * 100
    print(f"{'生成样本':<20} {gen_benign:>15} {gen_malicious:>15} {asr:>14.2f}%")
    
    dga_benign = (real_dga_preds == 0).sum().item()
    dga_malicious = (real_dga_preds == 1).sum().item()
    dga_acc = dga_malicious / len(real_dga_samples) * 100
    print(f"{'真实DGA':<20} {dga_benign:>15} {dga_malicious:>15} {dga_acc:>14.2f}%")
    
    ben_benign = (benign_preds == 0).sum().item()
    ben_malicious = (benign_preds == 1).sum().item()
    ben_acc = ben_benign / len(benign_samples) * 100
    print(f"{'真实良性':<20} {ben_benign:>15} {ben_malicious:>15} {ben_acc:>14.2f}%")
    
    # 置信度分析
    print(f"\n{'样本类型':<20} {'良性置信度':>15} {'恶意置信度':>15}")
    print("-"*70)
    print(f"{'生成样本':<20} {gen_probs[:, 0].mean().item():>15.4f} {gen_probs[:, 1].mean().item():>15.4f}")
    print(f"{'真实DGA':<20} {real_dga_probs[:, 0].mean().item():>15.4f} {real_dga_probs[:, 1].mean().item():>15.4f}")
    print(f"{'真实良性':<20} {benign_probs[:, 0].mean().item():>15.4f} {benign_probs[:, 1].mean().item():>15.4f}")
    
    # 总结与建议
    print(f"\n{'='*70}")
    print("📝 分析总结与建议")
    print("="*70)
    
    print(f"\n✅ 成功指标:")
    print(f"  - 攻击成功率 (ASR): {asr:.2f}%")
    print(f"  - 生成样本平均长度: {gen_stats['avg_length']:.1f} (真实DGA: {real_dga_stats['avg_length']:.1f})")
    
    print(f"\n⚠️ 发现的问题:")
    if gen_stats['avg_length'] < real_dga_stats['avg_length'] * 0.5:
        print(f"  - 生成序列过短，平均长度仅为真实DGA的 {gen_stats['avg_length']/real_dga_stats['avg_length']*100:.1f}%")
    if gen_stats['unique_chars'] < real_dga_stats['unique_chars']:
        print(f"  - 字符多样性不足，仅使用 {gen_stats['unique_chars']} 种字符 (真实: {real_dga_stats['unique_chars']})")
    if asr < 40:
        print(f"  - ASR较低 ({asr:.1f}%)，需要进一步优化")
    
    print(f"\n🎯 优化建议:")
    if asr < 50:
        print(f"  1. 继续训练至更多轮次 (当前可能仅10轮)")
        print(f"  2. 调整生成器架构，增加hidden_dim或增加LSTM层数")
        print(f"  3. 尝试不同的学习率 (当前1e-4)")
    if gen_stats['avg_length'] < real_dga_stats['avg_length'] * 0.7:
        print(f"  4. 在损失函数中加入长度惩罚，鼓励生成更长序列")
    
    # 相似度评估
    length_similarity = 1 - abs(gen_stats['avg_length'] - real_dga_stats['avg_length']) / real_dga_stats['avg_length']
    char_similarity = gen_stats['unique_chars'] / real_dga_stats['unique_chars']
    
    print(f"\n📈 与真实DGA相似度:")
    print(f"  - 长度相似度: {length_similarity*100:.2f}%")
    print(f"  - 字符多样性相似度: {char_similarity*100:.2f}%")
    print(f"  - 综合评分: {(length_similarity + char_similarity + asr/100)/3*100:.2f}%")

if __name__ == "__main__":
    main()
