#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba-MoE 模型训练脚本 - 极简正确版
Mamba层 + MoE层交替堆叠，Token级稀疏激活
"""

import torch
import argparse
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.model.mamba1_moe_model import LightweightMambaMoE
from core.logger import get_training_logger
from core.trainer import TrainerBuilder


def main(task_type: str = 'binary',
         dataset_path: str = '/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
         batch_size: int = 32,
         epochs: int = 10,
         learning_rate: float = 0.001,
         num_layers: int = 2,
         num_experts: int = 3,
         d_state: int = 16,
         balance_weight: float = 1e-2):
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 错误: Mamba-MoE 模型需要 CUDA 环境")
        return
    
    device = torch.device('cuda')
    logger = get_training_logger("mamba_moe", task_type)
    logger.info(f"开始 Mamba-MoE 模型{'二分类' if task_type == 'binary' else '多分类'}任务")
    logger.info(f"模型配置: num_layers={num_layers}, num_experts={num_experts}, d_state={d_state}")
    logger.info(f"损失权重: balance_weight={balance_weight}")
    
    # 1. 加载数据
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        task_type=task_type
    )
    print_dataset_info(dataset_info)
    
    # 2. 创建模型（权重在这里唯一定义）
    model = LightweightMambaMoE(
        vocab_size=dataset_info['vocab_size'],
        embedding_dim=128,
        max_length=dataset_info['max_length'],
        num_classes=dataset_info['num_classes'],
        num_layers=num_layers,
        d_state=d_state,
        d_conv=4,
        expand=2,
        num_experts=num_experts,
        expert_hidden=256,
        dropout_rate=0.3,
        balance_weight=balance_weight
    ).to(device)
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. 创建训练器（criterion仅作占位，实际使用model.compute_loss）
    trainer = TrainerBuilder(model, model_name='mamba_moe', task_type=task_type) \
        .with_criterion(torch.nn.CrossEntropyLoss()) \
        .with_optimizer(torch.optim.Adam, lr=learning_rate) \
        .with_scheduler(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5) \
        .with_device(device) \
        .with_logger(logger) \
        .build()
    
    # 4. 定义MoE专用训练循环
    def train_one_epoch_moe(loader, is_train=True):
        model.train() if is_train else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        expert_stats = torch.zeros(num_experts).to(device)
        
        num_batches = len(loader)
        for batch_idx, (data, targets) in enumerate(loader):
            data, targets = data.to(device), targets.to(device)
            
            if is_train:
                trainer.optimizer.zero_grad()
                # 关键：调用 model.compute_loss() 而非 criterion()
                logits, gate_info = model(data, return_gate_info=True)
                loss = model.compute_loss(logits, targets, gate_info)
                loss.backward()
                trainer.optimizer.step()
            else:
                with torch.no_grad():
                    logits, gate_info = model(data, return_gate_info=True)
                    # 验证时也计算总损失，保持可比性
                    loss = model.compute_loss(logits, targets, gate_info)
            
            total_loss += loss.item()
            correct += logits.max(1)[1].eq(targets).sum().item()
            total += targets.size(0)
            expert_stats += gate_info['expert_usage'].mean(dim=0)
            
            # 打印训练进度
            if is_train and (batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1):
                progress = (batch_idx + 1) / num_batches * 100
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100. * correct / total
                logger.info(f'Batch [{batch_idx+1}/{num_batches}], '
                           f'Progress: {progress:.1f}%, Loss: {loss.item():.4f}, '
                           f'Avg Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')
        
        return total_loss / len(loader), correct / total, expert_stats / len(loader)
    
    # 5. 训练循环
    best_acc, patience_counter = 0.0, 0
    for epoch in range(epochs):
        logger.info(f"\n{'='*50}\nEpoch [{epoch+1}/{epochs}]\n{'='*50}")
        
        # 训练
        train_loss, train_acc, train_usage = train_one_epoch_moe(train_loader, is_train=True)
        
        # 验证
        val_loss, val_acc, val_usage = train_one_epoch_moe(val_loader, is_train=False)
        
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        # 日志
        logger.info(f'Epoch [{epoch+1}/{epochs}] 完成')
        logger.info(f'Train: loss={train_loss:.4f} acc={train_acc:.4f} expert_usage={train_usage.tolist()}')
        logger.info(f'Val:   loss={val_loss:.4f} acc={val_acc:.4f} expert_usage={val_usage.tolist()}')
        
        # 早停
        if val_acc > best_acc:
            best_acc, patience_counter = val_acc, 0
            model_path = f'./models/mamba_moe_best_{task_type}.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            trainer.save_model(model_path)
            logger.info(f'新的最佳验证准确率: {best_acc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= 3:
                logger.info("早停触发")
                break
    
    # 6. 测试
    logger.info("\n在测试集上评估...")
    y_pred, y_true = trainer.predict(test_loader)
    eval_results = trainer.evaluate(y_true, y_pred, dataset_info['class_names'])
    
    logger.info("=" * 50)
    logger.info("最终测试结果:")
    logger.info(f"准确率: {eval_results['accuracy']:.4f}")
    logger.info(f"精确率: {eval_results['precision']:.4f}")
    logger.info(f"召回率: {eval_results['recall']:.4f}")
    logger.info(f"F1分数: {eval_results['f1_score']:.4f}")
    
    if 'class_metrics' in eval_results:
        logger.info("\n各类别详细指标:")
        for class_name, metrics in eval_results['class_metrics'].items():
            logger.info(f"{class_name}: "
                       f"Precision={metrics['precision']:.4f}, "
                       f"Recall={metrics['recall']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}")
    
    logger.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba-MoE 模型训练')
    parser.add_argument('--task', default='binary', choices=['binary', 'multiclass'])
    parser.add_argument('--dataset', default='/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--balance_weight', type=float, default=1e-2)
    args = parser.parse_args()
    
    main(task_type=args.task,
         dataset_path=args.dataset,
         batch_size=args.batch_size,
         epochs=args.epochs,
         learning_rate=args.lr,
         num_layers=args.num_layers,
         num_experts=args.num_experts,
         d_state=args.d_state,
         balance_weight=args.balance_weight)
