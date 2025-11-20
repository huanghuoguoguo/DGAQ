#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba2 模型训练脚本
基于 SSD 算法的 Mamba2 层，无 MoE 结构
支持二分类和多分类任务
"""

import torch
import argparse
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.model.mamba2_model import LightweightMamba2
from core.logger import get_training_logger
from core.trainer import TrainerBuilder


def main(task_type: str = 'binary',
         dataset_path: str = '/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
         batch_size: int = 32,
         epochs: int = 10,
         learning_rate: float = 0.001,
         num_layers: int = 2,
         d_state: int = 128,
         headdim: int = 64):
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 错误: Mamba2 模型需要 CUDA 环境")
        return
    
    device = torch.device('cuda')
    logger = get_training_logger("mamba2", task_type)
    logger.info(f"开始 Mamba2 模型{'二分类' if task_type == 'binary' else '多分类'}任务")
    logger.info(f"模型配置: num_layers={num_layers}, d_state={d_state}, headdim={headdim}")
    
    # 1. 加载数据
    logger.info("正在加载数据集...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        task_type=task_type
    )
    print_dataset_info(dataset_info)
    logger.info("数据集加载完成")
    
    # 2. 创建 Mamba2 模型
    logger.info("正在创建 Mamba2 模型...")
    model = LightweightMamba2(
        vocab_size=dataset_info['vocab_size'],
        embedding_dim=256,  # Mamba2需要较大的嵌入维度，以满足对齐要求
        max_length=dataset_info['max_length'],
        num_classes=dataset_info['num_classes'],
        num_layers=num_layers,
        d_state=d_state,
        d_conv=4,
        expand=2,
        headdim=headdim,
        dropout_rate=0.3
    ).to(device)
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"模型结构:\n{model}")
    
    # 3. 创建训练器
    logger.info("正在创建训练器...")
    trainer = TrainerBuilder(model, model_name='mamba2', task_type=task_type) \
        .with_criterion(torch.nn.CrossEntropyLoss()) \
        .with_optimizer(torch.optim.Adam, lr=learning_rate) \
        .with_scheduler(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5) \
        .with_device(device) \
        .with_logger(logger) \
        .build()
    
    # 4. 训练模型
    logger.info("开始训练模型...")
    
    # 训练回调函数 - 保存最佳模型
    best_acc = 0.0
    save_path = f'./models/mamba2_best_{task_type}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def on_epoch_end(epoch, train_metrics, val_metrics):
        nonlocal best_acc
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            trainer.save_model(save_path)
            logger.info(f'新的最佳验证准确率: {best_acc:.2f}%')
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        callbacks={'on_epoch_end': on_epoch_end}
    )
    
    # 5. 在测试集上评估
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
    parser = argparse.ArgumentParser(description='Mamba2 模型训练')
    parser.add_argument('--task', default='binary', choices=['binary', 'multiclass'],
                       help='任务类型: binary(二分类) 或 multiclass(多分类)')
    parser.add_argument('--dataset', default='/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
                       help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Mamba2 层数')
    parser.add_argument('--d_state', type=int, default=128,
                       help='SSD 状态维度')
    parser.add_argument('--headdim', type=int, default=64,
                       help='每个头的维度')
    args = parser.parse_args()
    
    print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    main(task_type=args.task,
         dataset_path=args.dataset,
         batch_size=args.batch_size,
         epochs=args.epochs,
         learning_rate=args.lr,
         num_layers=args.num_layers,
         d_state=args.d_state,
         headdim=args.headdim)
