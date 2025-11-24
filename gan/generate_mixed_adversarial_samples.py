#!/usr/bin/env python
"""生成混合多epoch的对抗样本用于对抗训练"""

import torch
import argparse
import os
import sys
import pickle
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.adversarial.generator import DGAGenerator
from core.dataset import load_dataset

def load_generator(model_path, device, config):
    """加载生成器"""
    generator = DGAGenerator(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        max_len=config['max_len'],
        z_dim=config['z_dim']
    ).to(device)
    
    if os.path.exists(model_path):
        print(f"加载生成器: {model_path}")
        generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    else:
        raise FileNotFoundError(f"生成器文件未找到: {model_path}")
    
    generator.eval()
    return generator

def generate_mixed_samples(args):
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
    
    gen_config = {
        'vocab_size': dataset_info.get('vocab_size', 41),
        'hidden_dim': args.hidden_dim,
        'max_len': dataset_info.get('max_length', 60),
        'z_dim': args.z_dim
    }
    
    # 2. 加载多个生成器并生成样本
    generator_epochs = args.epochs  # [5, 10, 20, 30]
    samples_per_epoch = args.num_samples // len(generator_epochs)
    
    print(f"\n混合生成对抗样本:")
    print(f"  总样本数: {args.num_samples}")
    print(f"  生成器epochs: {generator_epochs}")
    print(f"  每个epoch生成: {samples_per_epoch}")
    print()
    
    all_samples = []
    
    for epoch in generator_epochs:
        gen_path = f"./models/gan/generator_epoch_{epoch}.pth"
        
        if not os.path.exists(gen_path):
            print(f"⚠️  生成器 Epoch {epoch} 不存在，跳过")
            continue
        
        generator = load_generator(gen_path, device, gen_config)
        
        print(f"生成 Epoch {epoch} 样本: {samples_per_epoch}...")
        batch_size = args.batch_size
        num_batches = (samples_per_epoch + batch_size - 1) // batch_size
        
        epoch_samples = []
        with torch.no_grad():
            for i in range(num_batches):
                current_batch_size = min(batch_size, samples_per_epoch - i * batch_size)
                samples = generator.sample(current_batch_size, device)
                epoch_samples.append(samples.cpu().numpy())
        
        epoch_samples = np.concatenate(epoch_samples, axis=0)[:samples_per_epoch]
        all_samples.append(epoch_samples)
        print(f"  ✓ 完成，形状: {epoch_samples.shape}")
    
    # 3. 合并所有样本
    all_samples = np.concatenate(all_samples, axis=0)
    print(f"\n✓ 总生成完成，样本形状: {all_samples.shape}")
    
    # 4. 创建标签（全部为恶意=1）
    labels = np.ones(len(all_samples), dtype=np.int64)
    
    # 5. 保存
    adversarial_data = {
        'domains': all_samples,
        'labels': labels,
        'metadata': dataset_info
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(adversarial_data, f)
    
    print(f"\n✓ 混合对抗样本已保存到: {args.output_path}")
    print(f"  - 样本数量: {len(all_samples)}")
    print(f"  - 标签分布: 恶意={len(all_samples)}, 良性=0")
    
    # 6. 统计
    print("\n样本统计:")
    lengths = (all_samples != 0).sum(axis=1)
    print(f"  平均长度: {lengths.mean():.2f}")
    print(f"  最小长度: {lengths.min()}")
    print(f"  最大长度: {lengths.max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成混合多epoch的对抗样本")
    parser.add_argument('--dataset_path', type=str, 
                        default='./data/processed/500k_unified_dga_dataset.pkl')
    parser.add_argument('--output_path', type=str, 
                        default='./data/processed/mixed_adversarial_samples.pkl')
    parser.add_argument('--epochs', nargs='+', type=int, 
                        default=[5, 10, 20, 30],
                        help='要混合的生成器epochs')
    parser.add_argument('--num_samples', type=int, default=60000,
                        help='总生成样本数量')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    args = parser.parse_args()
    generate_mixed_samples(args)
