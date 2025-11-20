#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用通用训练器框架训练LSTM模型示例
展示如何将训练器应用到不同模型
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.dataset import create_data_loaders, print_dataset_info
from core.logger import get_training_logger
from core.trainer import TrainerBuilder


class SimpleLSTM(nn.Module):
    """简单的LSTM模型示例"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 128, num_classes: int = 2, dropout_rate: float = 0.5):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.dropout(x)
        x = self.fc(x)
        return x


def main(task_type='binary', dataset_path='/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
         batch_size=64, epochs=5, learning_rate=0.001):
    """主函数 - 演示通用训练器的使用"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置日志
    logger = get_training_logger("lstm", task_type)
    logger.info("开始LSTM模型训练演示")
    
    try:
        # 加载数据
        print("正在加载数据集...")
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=dataset_path,
            batch_size=batch_size,
            task_type=task_type
        )
        print_dataset_info(dataset_info)
        
        # 创建LSTM模型
        print("正在创建LSTM模型...")
        model = SimpleLSTM(
            vocab_size=dataset_info['vocab_size'],
            embedding_dim=128,
            hidden_dim=128,
            num_classes=dataset_info['num_classes'],
            dropout_rate=0.5
        )
        
        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # 使用通用训练器框架 - 链式调用
        print("正在创建训练器...")
        trainer = TrainerBuilder(model, model_name='lstm', task_type=task_type) \
            .with_criterion(nn.CrossEntropyLoss()) \
            .with_optimizer(torch.optim.Adam, lr=learning_rate) \
            .with_scheduler(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5) \
            .with_device(device) \
            .with_logger(logger) \
            .build()
        
        # 训练模型
        print("开始训练...")
        history = trainer.train(train_loader, val_loader, num_epochs=epochs)
        
        # 预测和评估
        print("正在测试集上进行预测...")
        y_pred, y_true = trainer.predict(test_loader)
        
        eval_results = trainer.evaluate(y_true, y_pred, dataset_info['class_names'])
        
        print("=" * 50)
        print("模型评估结果:")
        print(f"准确率: {eval_results['accuracy']:.4f}")
        print(f"精确率: {eval_results['precision']:.4f}")
        print(f"召回率: {eval_results['recall']:.4f}")
        print(f"F1分数: {eval_results['f1_score']:.4f}")
        print("=" * 50)
        
        # 保存模型
        model_path = "./models/lstm_demo_model.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        print("训练完成!")
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LSTM模型训练')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multiclass'],
                        help='任务类型: binary(二分类) 或 multiclass(多分类)')
    parser.add_argument('--dataset', type=str, default='/jsj_ywj/yhh/DGAQ/data/processed/500k_unified_dga_dataset.pkl',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
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