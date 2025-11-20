#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba 模型训练脚本
支持二分类和多分类任务
基于 State Space Model (SSM)，线性复杂度，适合长序列
"""

import torch
import argparse
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.model.mamba_model import LightweightMamba
from core.logger import get_training_logger
from core.trainer import TrainerBuilder


def main(task_type: str = 'binary',
         dataset_path: str = '/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
         batch_size: int = 32,
         epochs: int = 10,
         learning_rate: float = 0.001,
         num_layers: int = 2,
         d_state: int = 16,
         expand: int = 2):
    """
    主训练函数
    
    Args:
        task_type: 任务类型 ('binary' 或 'multiclass')
        dataset_path: 数据集路径
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        num_layers: Mamba 层数
        d_state: SSM 状态维度
        expand: 扩展系数
    """
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 错误: Mamba 模型需要 CUDA 环境")
        print("mamba-ssm 暂不支持 CPU 运行")
        return
    
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 设置日志记录器
    logger = get_training_logger("mamba", task_type)
    logger.info(f"开始 Mamba 模型{'二分类' if task_type == 'binary' else '多分类'}任务")
    logger.info(f"模型配置: num_layers={num_layers}, d_state={d_state}, expand={expand}")
    
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
        
        # 创建 Mamba 模型
        logger.info("正在创建 Mamba 模型...")
        model = LightweightMamba(
            vocab_size=vocab_size,
            embedding_dim=128,
            max_length=max_length,
            num_classes=num_classes,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=4,
            expand=expand,
            dropout_rate=0.3
        )
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数量: {total_params:,}")
        logger.info(f"模型结构:\n{model}")
        
        # 使用通用训练器框架
        logger.info("正在创建训练器...")
        trainer = TrainerBuilder(model, model_name='mamba', task_type=task_type) \
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
        model_path = f"./models/mamba_{task_type}_model.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        logger.info("训练和评估完成!")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Mamba 模型训练')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multiclass'],
                        help='任务类型: binary(二分类) 或 multiclass(多分类)')
    parser.add_argument('--dataset', type=str, 
                        default='/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_layers', type=int, default=2, help='Mamba 层数')
    parser.add_argument('--d_state', type=int, default=16, help='SSM 状态维度')
    parser.add_argument('--expand', type=int, default=2, help='扩展系数')
    
    args = parser.parse_args()
    
    # 运行主函数
    main(
        task_type=args.task,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_layers=args.num_layers,
        d_state=args.d_state,
        expand=args.expand
    )
