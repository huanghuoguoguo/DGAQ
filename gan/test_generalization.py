#!/usr/bin/env python
"""测试对抗训练模型的泛化能力：用不同epoch的生成器攻击"""

import os
import sys
import torch
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.model.cnn_model import LightweightCNN
from core.model.cnn_moe_model import LightweightCNNMoE
from core.adversarial.generator import DGAGenerator
from core.dataset import load_dataset

def load_target_model(model_path, device, dataset_info):
    """加载目标分类器"""
    if 'moe' in model_path.lower():
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        expert_keys = [k for k in state.keys() if k.startswith('experts.')]
        num_experts = 3
        if expert_keys:
            max_expert = max([int(k.split('.')[1]) for k in expert_keys])
            num_experts = max_expert + 1
        
        model = LightweightCNNMoE(
            vocab_size=dataset_info['vocab_size'],
            embedding_dim=128,
            max_length=dataset_info['max_length'],
            num_classes=dataset_info['num_classes'],
            num_experts=num_experts
        ).to(device)
    else:
        model = LightweightCNN(
            vocab_size=dataset_info['vocab_size'],
            embedding_dim=128,
            max_length=dataset_info['max_length'],
            num_classes=dataset_info['num_classes']
        ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model

def load_generator(model_path, device, config):
    """加载生成器"""
    generator = DGAGenerator(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        max_len=config['max_len'],
        z_dim=config['z_dim']
    ).to(device)
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    generator.eval()
    return generator

def evaluate_attack(generator, target_model, device, num_samples):
    """评估攻击效果"""
    with torch.no_grad():
        adv_indices = generator.sample(num_samples, device)
        logits = target_model(adv_indices)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        successful_evasions = (predictions == 0).sum().item()
        asr = successful_evasions / num_samples
        
        return {
            'asr': asr,
            'evasions': successful_evasions,
            'detections': num_samples - successful_evasions
        }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 配置
    dataset_path = './data/processed/500k_unified_dga_dataset.pkl'
    num_samples = 2000
    
    # 获取数据集信息
    try:
        from core.dataset import create_data_loaders
        _, _, _, dataset_info = create_data_loaders(dataset_path, batch_size=32, task_type='binary')
    except:
        dataset_info = {'vocab_size': 41, 'max_length': 60, 'num_classes': 2}
    
    gen_config = {
        'vocab_size': dataset_info.get('vocab_size', 41),
        'hidden_dim': 256,
        'max_len': dataset_info.get('max_length', 60),
        'z_dim': 100
    }
    
    # 测试的模型
    models = [
        ("原始CNN", "./models/cnn_binary_model.pth"),
        ("对抗训练CNN", "./models/cnn_adversarial_trained.pth"),
        ("原始CNN-MoE", "./models/cnn_moe_binary_model.pth"),
        ("对抗训练CNN-MoE", "./models/cnn_moe_adversarial_trained.pth"),
    ]
    
    # 测试的生成器（不同epoch）
    generator_epochs = [5, 10, 20, 30, 40, 50]
    
    print("="*80)
    print("泛化能力测试：用不同epoch的生成器攻击对抗训练前后的模型")
    print("="*80)
    print()
    
    results = {}
    
    for model_name, model_path in models:
        if not os.path.exists(model_path):
            print(f"⚠️  跳过 {model_name}（模型不存在）")
            continue
        
        print(f"\n{'='*80}")
        print(f"目标模型: {model_name}")
        print(f"{'='*80}")
        
        target_model = load_target_model(model_path, device, dataset_info)
        results[model_name] = {}
        
        for epoch in generator_epochs:
            gen_path = f"./models/gan/generator_epoch_{epoch}.pth"
            
            if not os.path.exists(gen_path):
                print(f"  Epoch {epoch:2d}: 生成器不存在，跳过")
                continue
            
            generator = load_generator(gen_path, device, gen_config)
            metrics = evaluate_attack(generator, target_model, device, num_samples)
            
            results[model_name][epoch] = metrics
            
            print(f"  Epoch {epoch:2d}: ASR={metrics['asr']*100:5.2f}%, "
                  f"逃逸={metrics['evasions']:4d}, 检测={metrics['detections']:4d}")
    
    # 生成对比分析
    print(f"\n\n{'='*80}")
    print("📊 泛化能力分析")
    print(f"{'='*80}\n")
    
    # 对比表格
    print(f"{'生成器Epoch':<15}", end='')
    for model_name in ["原始CNN", "对抗训练CNN", "原始CNN-MoE", "对抗训练CNN-MoE"]:
        if model_name in results:
            print(f"{model_name:>20}", end='')
    print()
    print("-"*80)
    
    for epoch in generator_epochs:
        print(f"Epoch {epoch:<8}", end='')
        for model_name in ["原始CNN", "对抗训练CNN", "原始CNN-MoE", "对抗训练CNN-MoE"]:
            if model_name in results and epoch in results[model_name]:
                asr = results[model_name][epoch]['asr'] * 100
                print(f"{asr:>19.2f}%", end='')
            else:
                print(f"{'N/A':>20}", end='')
        print()
    
    # 关键发现
    print(f"\n{'='*80}")
    print("🔍 关键发现")
    print(f"{'='*80}\n")
    
    if "原始CNN" in results and "对抗训练CNN" in results:
        print("CNN模型对比：")
        for epoch in generator_epochs:
            if epoch in results["原始CNN"] and epoch in results["对抗训练CNN"]:
                original_asr = results["原始CNN"][epoch]['asr'] * 100
                trained_asr = results["对抗训练CNN"][epoch]['asr'] * 100
                reduction = original_asr - trained_asr
                print(f"  Epoch {epoch}: {original_asr:.1f}% → {trained_asr:.1f}% "
                      f"(降低 {reduction:.1f}%)")
        
        # 计算平均泛化能力
        avg_reduction = sum([
            results["原始CNN"][e]['asr'] - results["对抗训练CNN"][e]['asr']
            for e in generator_epochs
            if e in results["原始CNN"] and e in results["对抗训练CNN"]
        ]) / len([e for e in generator_epochs 
                  if e in results["原始CNN"] and e in results["对抗训练CNN"]])
        
        print(f"\n  平均ASR降低: {avg_reduction*100:.1f}%")
        print(f"  对抗训练效果: {'✅ 显著' if avg_reduction > 0.3 else '⚠️  一般' if avg_reduction > 0.1 else '❌ 较弱'}")

if __name__ == "__main__":
    main()
