#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 统一数据集处理模块
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Any


class DGADataset(Dataset):
    """统一的DGA数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(file_path: str = './data/processed/small_dga_dataset.pkl') -> Dict[str, Any]:
    """加载预处理的数据集"""
    if not os.path.exists(file_path):
        # 尝试原始位置
        fallback_path = './data/small_dga_dataset.pkl'
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def create_data_loaders(dataset_path: str = './data/processed/small_dga_dataset.pkl',
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       random_seed: int = 42,
                       task_type: str = 'binary') -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """创建训练、验证、测试数据加载器"""
    
    # 加载数据集
    dataset = load_dataset(dataset_path)
    
    # 检查数据集结构类型
    if 'train' in dataset and 'val' in dataset and 'test' in dataset:
        # 统一数据集格式
        train_data = dataset['train']
        val_data = dataset['val']
        test_data = dataset['test']
        
        # 检查是否有info或metadata
        if 'info' in dataset:
            metadata = dataset['info']
        elif 'metadata' in dataset:
            metadata = dataset['metadata']
        else:
            raise KeyError("数据集缺少metadata信息")
        
        # 处理域名序列化
        if 'sequences' in train_data:
            # 已序列化的数据
            X_train = np.array(train_data['sequences'])
            X_val = np.array(val_data['sequences'])
            X_test = np.array(test_data['sequences'])
        elif 'domains' in train_data:
            # 需要序列化的域名数据
            # 创建字符映射
            chars = set()
            for domains in [train_data['domains'], val_data['domains'], test_data['domains']]:
                for domain in domains:
                    chars.update(domain.lower())
            
            # 构建字符到索引的映射
            char_to_idx = {'<PAD>': 0, '<UNK>': 1}
            for i, char in enumerate(sorted(chars), 2):
                char_to_idx[char] = i
            
            def domain_to_sequence(domain: str, max_len: int = 60):
                sequence = [char_to_idx.get(char.lower(), char_to_idx['<UNK>']) for char in domain]
                if len(sequence) > max_len:
                    sequence = sequence[:max_len]
                else:
                    sequence.extend([char_to_idx['<PAD>']] * (max_len - len(sequence)))
                return sequence
            
            max_length = 60  # 默认最大长度
            X_train = np.array([domain_to_sequence(domain, max_length) for domain in train_data['domains']])
            X_val = np.array([domain_to_sequence(domain, max_length) for domain in val_data['domains']])
            X_test = np.array([domain_to_sequence(domain, max_length) for domain in test_data['domains']])
            
            # 更新词汇表大小
            vocab_size = len(char_to_idx)
        else:
            raise KeyError("数据集缺少域名数据")
        
        # 转换标签为numpy数组
        y_train = np.array(train_data['labels'])
        y_val = np.array(val_data['labels'])
        y_test = np.array(test_data['labels'])
        
        # 根据任务类型转换标签
        if task_type == 'binary':
            # 二分类：0=良性，1=恶意（所有非0标签转为1）
            y_train = (y_train > 0).astype(int)
            y_val = (y_val > 0).astype(int)
            y_test = (y_test > 0).astype(int)
            num_classes = 2
            class_names = ['benign', 'malicious']
        else:
            # 多分类：保持原始标签，类别数应该是最大标签值+1
            all_labels = np.concatenate([y_train, y_val, y_test])
            num_classes = int(np.max(all_labels)) + 1
            
            # 尝试从metadata中获取真实的类别名称
            if 'label_mapping' in metadata:
                # 使用label_mapping构建class_names
                label_to_name = {v: k for k, v in metadata['label_mapping'].items()}
                class_names = [label_to_name.get(i, f'unknown_{i}') for i in range(num_classes)]
            elif 'malicious_families' in metadata:
                # 使用malicious_families构建class_names
                class_names = ['benign'] + metadata['malicious_families'][:num_classes-1]
            else:
                # 降级到通用命名
                class_names = ['benign'] + [f'malicious_{i}' for i in range(1, num_classes)]
        
        # 创建数据集对象
        train_dataset = DGADataset(X_train, y_train)
        val_dataset = DGADataset(X_val, y_val)
        test_dataset = DGADataset(X_test, y_test)
        
        # 数据集信息
        dataset_info = {
            'vocab_size': vocab_size if 'vocab_size' in locals() else metadata.get('vocab_size', 128),
            'max_length': metadata.get('max_length', X_train.shape[1]),
            'num_classes': num_classes,
            'class_names': class_names,
            'total_samples': len(X_train) + len(X_val) + len(X_test),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'task_type': task_type,
            'class_distribution': {
                'train': np.bincount(y_train),
                'val': np.bincount(y_val),
                'test': np.bincount(y_test)
            }
        }
        
    elif 'X_train' in dataset:
        # 旧格式的多分类数据集
        X_train, y_train = dataset['X_train'], dataset['y_train']
        X_val, y_val = dataset['X_val'], dataset['y_val']
        X_test, y_test = dataset['X_test'], dataset['y_test']
        
        # 创建数据集对象
        train_dataset = DGADataset(X_train, y_train)
        val_dataset = DGADataset(X_val, y_val)
        test_dataset = DGADataset(X_test, y_test)
        
        # 数据集信息
        dataset_info = {
            'vocab_size': dataset['vocab_size'],
            'max_length': dataset.get('max_length', X_train.shape[1]),
            'num_classes': dataset.get('num_classes', 2),
            'class_names': dataset.get('class_names', ['benign', 'malicious']),
            'total_samples': len(X_train) + len(X_val) + len(X_test),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'class_distribution': np.bincount(y_train)
        }
        
    else:
        # 旧格式数据集需要分割
        X, y = dataset['X'], dataset['y']
        
        # 根据任务类型转换标签
        if task_type == 'binary':
            # 二分类：0=良性，1=恶意（所有非0标签转为1）
            y = (y > 0).astype(int)
            num_classes = 2
            class_names = ['benign', 'malicious']
        else:
            # 多分类：保持原始标签
            num_classes = len(np.unique(y))
            class_names = ['benign'] + [f'malicious_{i}' for i in range(1, num_classes)]
        
        # 创建数据集对象
        full_dataset = DGADataset(X, y)
        
        # 计算划分大小
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # 数据集划分
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        # 数据集信息
        dataset_info = {
            'vocab_size': dataset['vocab_size'],
            'max_length': dataset.get('max_length', X.shape[1]),
            'num_classes': num_classes,
            'class_names': class_names,
            'total_samples': total_size,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'task_type': task_type,
            'class_distribution': np.bincount(y)
        }
    
    # 创建数据加载器 - 优化GPU利用率
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # 多进程数据加载
        pin_memory=True,  # 内存固定，加速GPU传输
        persistent_workers=True  # 保持工作进程活跃
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader, dataset_info


def print_dataset_info(dataset_info: Dict[str, Any]):
    """打印数据集信息"""
    print(f"📊 数据集信息:")
    print(f"  总样本数: {dataset_info['total_samples']}")
    print(f"  词汇表大小: {dataset_info['vocab_size']}")
    print(f"  最大序列长度: {dataset_info['max_length']}")
    print(f"  类别数: {dataset_info['num_classes']}")
    print(f"  类别名称: {dataset_info['class_names'][:5]}... (共{len(dataset_info['class_names'])}个)")
    print(f"  类别分布: {dataset_info['class_distribution']}")
    print(f"  训练集: {dataset_info['train_samples']}")
    print(f"  验证集: {dataset_info['val_samples']}")
    print(f"  测试集: {dataset_info['test_samples']}")


if __name__ == "__main__":
    # 测试加载500k数据集并检查类别名称
    print("测试加载500k数据集...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path='./data/processed/500k_unified_dga_dataset.pkl',
        batch_size=32,
        task_type='multiclass'
    )
    
    print(f"\n✅ 类别总数: {dataset_info['num_classes']}")
    print(f"\n🎯 所有类别名称:")
    for i, name in enumerate(dataset_info['class_names']):
        print(f"  {i:2d}: {name}")