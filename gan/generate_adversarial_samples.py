#!/usr/bin/env python
"""生成对抗样本用于对抗训练"""

import torch
import argparse
import os
import sys
import pickle
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.adversarial.generator import DGAGenerator
from core.dataset import load_dataset

def load_generator(model_path, device, config):
    """加载GAN生成器"""
    generator = DGAGenerator(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        max_len=config['max_len'],
        z_dim=config['z_dim']
    ).to(device)
    
    if os.path.exists(model_path):
        print(f"加载生成器: {model_path}")
        generator.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"生成器文件未找到: {model_path}")
    
    generator.eval()
    return generator

def generate_samples(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 获取数据集信息
    try:
        dataset = load_dataset(args.dataset_path)
        if 'info' in dataset:
            dataset_info = dataset['info']
        elif 'metadata' in dataset:
            dataset_info = dataset['metadata']
            if 'vocab_size' not in dataset_info:
                from core.dataset import create_data_loaders
                _, _, _, info = create_data_loaders(args.dataset_path, batch_size=32, task_type='binary')
                dataset_info = info
        else:
            from core.dataset import create_data_loaders
            _, _, _, dataset_info = create_data_loaders(args.dataset_path, batch_size=32, task_type='binary')
    except FileNotFoundError:
        print(f"数据集未找到: {args.dataset_path}，使用默认配置")
        dataset_info = {
            'vocab_size': 41,
            'max_length': 60,
            'num_classes': 2
        }
    
    print(f"数据集信息: Vocab={dataset_info.get('vocab_size')}, MaxLen={dataset_info.get('max_length')}")
    
    # 2. 加载生成器
    gen_config = {
        'vocab_size': dataset_info.get('vocab_size', 41),
        'hidden_dim': args.hidden_dim,
        'max_len': dataset_info.get('max_length', 60),
        'z_dim': args.z_dim
    }
    generator = load_generator(args.generator_path, device, gen_config)
    
    # 3. 生成对抗样本
    print(f"\n生成 {args.num_samples} 个对抗样本...")
    all_samples = []
    batch_size = args.batch_size
    num_batches = (args.num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, args.num_samples - i * batch_size)
            samples = generator.sample(current_batch_size, device)  # (B, L)
            all_samples.append(samples.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {(i + 1) * batch_size}/{args.num_samples}")
    
    all_samples = np.concatenate(all_samples, axis=0)[:args.num_samples]
    print(f"✓ 生成完成，样本形状: {all_samples.shape}")
    
    # 4. 创建标签（全部为恶意=1）
    labels = np.ones(args.num_samples, dtype=np.int64)
    
    # 5. 保存为pickle格式（与训练数据格式一致）
    adversarial_data = {
        'domains': all_samples,  # (N, max_len) 索引序列
        'labels': labels,        # (N,) 全部为1（恶意）
        'metadata': dataset_info
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(adversarial_data, f)
    
    print(f"\n✓ 对抗样本已保存到: {args.output_path}")
    print(f"  - 样本数量: {args.num_samples}")
    print(f"  - 标签分布: 恶意={args.num_samples}, 良性=0")
    print(f"  - 数据形状: domains={all_samples.shape}, labels={labels.shape}")
    
    # 6. 显示统计信息
    print("\n样本统计:")
    lengths = (all_samples != 0).sum(axis=1)
    print(f"  平均长度: {lengths.mean():.2f}")
    print(f"  最小长度: {lengths.min()}")
    print(f"  最大长度: {lengths.max()}")
    
    unique_chars = []
    for sample in all_samples[:100]:  # 采样前100个
        unique_chars.append(len(np.unique(sample[sample != 0])))
    print(f"  平均唯一字符数(前100样本): {np.mean(unique_chars):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成对抗样本用于对抗训练")
    parser.add_argument('--generator_path', type=str, 
                        default='./models/gan/generator_epoch_5.pth',
                        help='生成器模型路径')
    parser.add_argument('--dataset_path', type=str, 
                        default='./data/processed/500k_unified_dga_dataset.pkl',
                        help='原始数据集路径（用于获取vocab等信息）')
    parser.add_argument('--output_path', type=str, 
                        default='./data/processed/adversarial_samples.pkl',
                        help='对抗样本保存路径')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='生成样本数量（建议原始训练集的20-30%）')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='生成批次大小')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='噪声维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='生成器隐藏层维度')
    
    args = parser.parse_args()
    generate_samples(args)
