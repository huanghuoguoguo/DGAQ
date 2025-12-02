#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAN对抗攻击性能分析脚本
测试不同epoch的生成器对多个目标模型的攻击效果
"""

import os
import sys
import torch
import glob
import argparse
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.model.cnn_model import LightweightCNN
from core.adversarial.generator import DGAGenerator
from core.dataset import load_dataset

def load_target_model(model_path, device, vocab_size=41, num_classes=2):
    """加载目标分类器，支持CNN与CNN-MoE"""
    model = None
    try:
        if 'cnn_moe' in os.path.basename(model_path):
            from core.model.cnn_moe_model import LightweightCNNMoE
            model = LightweightCNNMoE(
                vocab_size=vocab_size,
                embedding_dim=128,
                max_length=60,
                num_classes=num_classes,
                num_experts=3
            ).to(device)
        else:
            from core.model.cnn_model import LightweightCNN
            model = LightweightCNN(
                vocab_size=vocab_size,
                embedding_dim=128,
                max_length=60,
                num_classes=num_classes
            ).to(device)
        
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    except Exception as e:
        raise e

def load_generator(model_path, device, vocab_size=41, hidden_dim=256, max_len=60, z_dim=100):
    """加载生成器"""
    generator = DGAGenerator(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        max_len=max_len,
        z_dim=z_dim
    ).to(device)
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        generator.eval()
        return generator
    else:
        raise FileNotFoundError(f"Generator not found: {model_path}")

def evaluate_attack(generator, target_model, device, num_samples=1000):
    """评估攻击效果"""
    with torch.no_grad():
        # 生成对抗样本
        adv_indices = generator.sample(num_samples, device)
        
        # 目标模型预测
        logits = target_model(adv_indices)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        # 计算ASR (假设0=良性，1=恶意，我们希望生成的恶意样本被误分类为良性)
        successful_evasions = (predictions == 0).sum().item()
        asr = successful_evasions / num_samples
        
        # 计算置信度统计
        benign_confidence = probs[:, 0].mean().item()
        malicious_confidence = probs[:, 1].mean().item()
        
        return {
            'asr': asr,
            'evasions': successful_evasions,
            'detections': num_samples - successful_evasions,
            'benign_conf': benign_confidence,
            'malicious_conf': malicious_confidence
        }

def main():
    parser = argparse.ArgumentParser(description="分析GAN攻击性能")
    parser.add_argument('--gan_dir', type=str, default='./models/gan', help='GAN模型目录')
    parser.add_argument('--target_models', nargs='+', 
                       default=['./models/cnn_binary_model.pth'],
                       help='目标模型路径列表')
    parser.add_argument('--num_samples', type=int, default=1000, help='测试样本数')
    parser.add_argument('--epochs_to_test', nargs='+', type=int,
                       default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                       help='要测试的epoch列表')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"测试样本数: {args.num_samples}")
    print("="*60)
    
    # 获取所有生成器checkpoint
    generator_files = sorted(glob.glob(os.path.join(args.gan_dir, 'generator_epoch_*.pth')))
    
    # 筛选指定epoch
    generators_to_test = []
    for epoch in args.epochs_to_test:
        gen_path = os.path.join(args.gan_dir, f'generator_epoch_{epoch}.pth')
        if os.path.exists(gen_path):
            generators_to_test.append((epoch, gen_path))
    
    if not generators_to_test:
        print("❌ 未找到任何生成器checkpoint")
        return
    
    print(f"找到 {len(generators_to_test)} 个生成器checkpoint")
    print(f"目标模型数: {len(args.target_models)}\n")
    
    # 存储结果
    results = {}
    
    # 对每个目标模型进行测试
    for target_path in args.target_models:
        model_name = os.path.basename(target_path).replace('.pth', '')
        print(f"\n{'='*60}")
        print(f"目标模型: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(target_path):
            print(f"⚠️ 跳过（模型不存在）: {target_path}")
            continue
        
        # 加载目标模型
        target_model = load_target_model(target_path, device)
        results[model_name] = []
        
        # 测试每个epoch的生成器
        for epoch, gen_path in tqdm(generators_to_test, desc=f"测试 {model_name}"):
            try:
                generator = load_generator(gen_path, device)
                metrics = evaluate_attack(generator, target_model, device, args.num_samples)
                
                results[model_name].append({
                    'epoch': epoch,
                    **metrics
                })
                
                print(f"  Epoch {epoch:2d}: ASR={metrics['asr']*100:5.2f}%, "
                      f"逃逸={metrics['evasions']:4d}, "
                      f"检测={metrics['detections']:4d}, "
                      f"良性置信度={metrics['benign_conf']:.3f}")
                
            except Exception as e:
                print(f"  Epoch {epoch:2d}: 错误 - {e}")
    
    # 生成分析报告
    print(f"\n\n{'='*60}")
    print("📊 综合分析报告")
    print(f"{'='*60}\n")
    
    for model_name, model_results in results.items():
        if not model_results:
            continue
        
        print(f"目标模型: {model_name}")
        print("-" * 60)
        
        # 找到最佳ASR
        best_result = max(model_results, key=lambda x: x['asr'])
        worst_result = min(model_results, key=lambda x: x['asr'])
        
        print(f"  最佳 ASR: {best_result['asr']*100:.2f}% (Epoch {best_result['epoch']})")
        print(f"  最差 ASR: {worst_result['asr']*100:.2f}% (Epoch {worst_result['epoch']})")
        
        # ASR趋势分析
        asrs = [r['asr'] for r in model_results]
        epochs = [r['epoch'] for r in model_results]
        
        if len(asrs) > 1:
            # 计算ASR增长率
            asr_growth = (asrs[-1] - asrs[0]) / asrs[0] * 100 if asrs[0] > 0 else 0
            print(f"  ASR增长率: {asr_growth:+.2f}% (Epoch {epochs[0]} → {epochs[-1]})")
            
            # 判断收敛情况
            if len(asrs) >= 3:
                last_3_var = sum((asrs[i] - asrs[i-1])**2 for i in range(-3, 0)) / 3
                if last_3_var < 0.001:
                    print(f"  收敛状态: ✅ 已收敛 (最后3轮方差={last_3_var:.6f})")
                else:
                    print(f"  收敛状态: 🔄 仍在优化 (最后3轮方差={last_3_var:.6f})")
        
        print()
    
    # 给出下一步建议
    print(f"\n{'='*60}")
    print("🎯 下一步建议")
    print(f"{'='*60}\n")
    
    for model_name, model_results in results.items():
        if not model_results or len(model_results) < 2:
            continue
        
        asrs = [r['asr'] for r in model_results]
        epochs = [r['epoch'] for r in model_results]
        
        print(f"{model_name}:")
        
        # 建议1: 训练轮数
        if asrs[-1] > asrs[-2]:
            print(f"  ✅ ASR仍在上升，建议继续训练至 Epoch {epochs[-1] + 20}")
        else:
            print(f"  ⚠️ ASR下降，可能过拟合，建议使用 Epoch {epochs[asrs.index(max(asrs))]} 的模型")
        
        # 建议2: ASR水平评估
        best_asr = max(asrs)
        if best_asr < 0.3:
            print(f"  📈 当前最佳ASR={best_asr*100:.1f}%较低，建议:")
            print(f"     - 调整学习率 (当前1e-4，可尝试5e-5或2e-4)")
            print(f"     - 增加生成器hidden_dim (当前256，可尝试512)")
            print(f"     - 调整n_critic比例 (当前5，可尝试3或7)")
        elif best_asr < 0.5:
            print(f"  🎯 当前最佳ASR={best_asr*100:.1f}%中等，继续优化有潜力")
        else:
            print(f"  🎉 当前最佳ASR={best_asr*100:.1f}%优秀，攻击效果显著")
        
        # 建议3: 迁移性测试
        if len(args.target_models) == 1:
            print(f"  🔬 建议测试对其他模型的迁移攻击能力:")
            print(f"     - Mamba2: ./models/mamba2_binary_model.pth")
            print(f"     - CNN-MoE: ./models/cnn_moe_binary_model.pth")
        
        print()

if __name__ == "__main__":
    main()
