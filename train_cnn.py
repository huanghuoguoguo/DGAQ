#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN模型训练和预测脚本
支持二分类和多分类任务
"""

import torch
import argparse
import os
import sys
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.dataset import create_data_loaders, print_dataset_info
from core.cnn_model import LightweightCNN
from core.logger import get_training_logger
from core.trainer import TrainerBuilder


def main(task_type: str = 'binary', dataset_path: str = './data/processed/small_dga_dataset.pkl',
         batch_size: int = 32, epochs: int = 10, learning_rate: float = 0.001):
    """
    主训练函数
    
    Args:
        task_type: 任务类型 ('binary' 或 'multiclass')
        dataset_path: 数据集路径
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置日志记录器
    logger = get_training_logger("cnn", task_type)
    logger.info(f"开始CNN模型{'二分类' if task_type == 'binary' else '多分类'}任务")
    
    try:
        # 创建数据加载器
        logger.info("正在加载数据集...")
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=dataset_path,
            batch_size=batch_size,
            task_type=task_type
        )
        
        # 打印数据集信息
        print_dataset_info(dataset_info)
        logger.info("数据集加载完成")
        
        # 获取数据集参数
        vocab_size = dataset_info['vocab_size']
        max_length = dataset_info['max_length']
        num_classes = dataset_info['num_classes']
        class_names = dataset_info['class_names']
        
        # 创建CNN模型
        logger.info("正在创建CNN模型...")
        model = LightweightCNN(
            vocab_size=vocab_size,
            embedding_dim=128,
            max_length=max_length,
            num_classes=num_classes,
            dropout_rate=0.5
        )
        
        # 打印模型信息
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        logger.info(f"模型结构:\n{model}")
        
        # 使用通用训练器框架
        logger.info("正在创建训练器...")
        trainer = TrainerBuilder(model, model_name='cnn', task_type=task_type) \
            .with_criterion(torch.nn.CrossEntropyLoss()) \
            .with_optimizer(torch.optim.Adam, lr=learning_rate) \
            .with_scheduler(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5) \
            .with_device(device) \
            .with_logger(logger) \
            .build()
        
        # 训练模型
        logger.info("开始训练模型...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs
        )
        
        # 在测试集上进行预测
        logger.info("正在测试集上进行预测...")
        y_pred, y_true = trainer.predict(test_loader)
        
        # 评估预测结果
        logger.info("正在评估预测结果...")
        eval_results = trainer.evaluate(y_true, y_pred, class_names)
        
        # 打印评估结果
        logger.info("=" * 50)
        logger.info("模型评估结果:")
        logger.info(f"准确率: {eval_results['accuracy']:.4f}")
        logger.info(f"精确率: {eval_results['precision']:.4f}")
        logger.info(f"召回率: {eval_results['recall']:.4f}")
        logger.info(f"F1分数: {eval_results['f1_score']:.4f}")
        
        # 如果有多分类任务的结果，打印每个类别的指标
        if 'class_metrics' in eval_results:
            logger.info("\n各类别详细指标:")
            for class_name, metrics in eval_results['class_metrics'].items():
                logger.info(f"{class_name}: "
                           f"Precision={metrics['precision']:.4f}, "
                           f"Recall={metrics['recall']:.4f}, "
                           f"F1={metrics['f1_score']:.4f}")
        
        logger.info("=" * 50)
        
        # 保存模型
        model_path = f"./models/cnn_{task_type}_model.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        logger.info("训练和评估完成!")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CNN模型训练和预测')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multiclass'],
                        help='任务类型: binary(二分类) 或 multiclass(多分类)')
                        
    parser.add_argument('--dataset', type=str, default='/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    # 运行主函数
    main(
        task_type=args.task,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )