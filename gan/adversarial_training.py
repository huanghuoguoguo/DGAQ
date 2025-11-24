#!/usr/bin/env python
"""对抗训练：混合真实数据和GAN生成的对抗样本"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import argparse
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.model.cnn_model import LightweightCNN
from core.model.cnn_moe_model import LightweightCNNMoE
from core.dataset import load_dataset, DGADataset, create_data_loaders

def load_adversarial_data(adv_path, original_dataset_info):
    """加载对抗样本数据"""
    with open(adv_path, 'rb') as f:
        adv_data = pickle.load(f)
    
    print(f"✓ 加载对抗样本: {adv_path}")
    print(f"  - 样本数量: {len(adv_data['labels'])}")
    print(f"  - 标签分布: {np.bincount(adv_data['labels'])}")
    
    # 创建Dataset
    adv_dataset = DGADataset(
        adv_data['domains'],
        adv_data['labels']
    )
    
    return adv_dataset

def create_mixed_dataloader(original_dataset_path, adversarial_path, batch_size, adv_ratio=0.3):
    """创建混合数据加载器
    
    Args:
        original_dataset_path: 原始数据集路径
        adversarial_path: 对抗样本路径
        batch_size: 批次大小
        adv_ratio: 对抗样本占比（0-1之间）
    """
    # 1. 使用create_data_loaders加载原始数据集（会自动处理字符串到索引的转换）
    train_loader_orig, val_loader, test_loader, dataset_info = create_data_loaders(
        original_dataset_path,
        batch_size=batch_size,
        task_type='binary'
    )
    
    # 2. 获取原始训练集Dataset（从DataLoader提取）
    train_dataset_orig = train_loader_orig.dataset
    
    # 3. 加载对抗样本（已经是索引序列）
    adv_dataset = load_adversarial_data(adversarial_path, dataset_info)
    
    # 4. 计算混合比例
    original_size = len(train_dataset_orig)
    adv_size = len(adv_dataset)
    
    # 根据adv_ratio调整对抗样本数量
    target_adv_size = int(original_size * adv_ratio / (1 - adv_ratio))
    
    if adv_size > target_adv_size:
        # 随机采样对抗样本
        indices = np.random.choice(adv_size, target_adv_size, replace=False)
        adv_subset = torch.utils.data.Subset(adv_dataset, indices)
        print(f"\n对抗样本采样: {adv_size} → {target_adv_size}")
    else:
        adv_subset = adv_dataset
        print(f"\n使用全部对抗样本: {adv_size}")
    
    # 5. 混合数据集
    mixed_dataset = ConcatDataset([train_dataset_orig, adv_subset])
    
    print(f"\n混合数据集:")
    print(f"  - 原始训练集: {len(train_dataset_orig)}")
    print(f"  - 对抗样本: {len(adv_subset)}")
    print(f"  - 总计: {len(mixed_dataset)}")
    print(f"  - 对抗样本比例: {len(adv_subset)/len(mixed_dataset)*100:.1f}%")
    
    # 6. 创建DataLoader
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset_info

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels

def train_adversarial(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 创建混合数据加载器
    train_loader, val_loader, test_loader, dataset_info = create_mixed_dataloader(
        args.dataset_path,
        args.adversarial_path,
        args.batch_size,
        args.adv_ratio
    )
    
    # 2. 创建模型
    if args.model_type == 'cnn':
        model = LightweightCNN(
            vocab_size=dataset_info['vocab_size'],
            embedding_dim=128,
            max_length=dataset_info['max_length'],
            num_classes=dataset_info['num_classes']
        ).to(device)
    elif args.model_type == 'cnn_moe':
        # 先尝试加载预训练模型以确定专家数量
        num_experts = 3  # 默认值
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            state = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
            expert_keys = [k for k in state.keys() if k.startswith('experts.')]
            if expert_keys:
                max_expert = max([int(k.split('.')[1]) for k in expert_keys])
                num_experts = max_expert + 1
                print(f"从预训练模型检测到专家数量: {num_experts}")
        
        model = LightweightCNNMoE(
            vocab_size=dataset_info['vocab_size'],
            embedding_dim=128,
            max_length=dataset_info['max_length'],
            num_classes=dataset_info['num_classes'],
            num_experts=num_experts
        ).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 3. 加载预训练模型（可选）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"\n加载预训练模型: {args.pretrained_path}")
        if args.model_type == 'cnn_moe':
            # CNN-MoE已经在上面加载过state了
            state = torch.load(args.pretrained_path, map_location=device, weights_only=False)
            model.load_state_dict(state)
        else:
            model.load_state_dict(torch.load(args.pretrained_path, map_location=device, weights_only=False))
    
    print(f"\n模型: {args.model_type}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    # 5. 训练循环
    best_val_acc = 0
    print(f"\n开始对抗训练 (Epochs={args.epochs})...")
    print("="*60)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100*correct/total:.2f}%'})
        
        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
    
    # 6. 测试集评估
    print("\n" + "="*60)
    print("最终测试集评估:")
    model.load_state_dict(torch.load(args.output_path, map_location=device))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"  Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    
    # 计算更多指标
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Benign', 'Malicious']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, test_preds))
    
    print(f"\n✓ 对抗训练完成，模型已保存: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对抗训练")
    parser.add_argument('--dataset_path', type=str, 
                        default='./data/processed/500k_unified_dga_dataset.pkl')
    parser.add_argument('--adversarial_path', type=str, 
                        default='./data/processed/adversarial_samples.pkl')
    parser.add_argument('--model_type', type=str, default='cnn', 
                        choices=['cnn', 'cnn_moe'])
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='预训练模型路径（可选）')
    parser.add_argument('--output_path', type=str, 
                        default='./models/cnn_adversarial_trained.pth')
    parser.add_argument('--adv_ratio', type=float, default=0.25,
                        help='对抗样本占比（0-1）')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    train_adversarial(args)
