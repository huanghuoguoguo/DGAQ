#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级CNN模型用于DGA检测
支持二分类和多分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any


class LightweightCNN(nn.Module):
    """轻量级CNN模型用于DGA检测"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, max_length: int = 60, 
                 num_classes: int = 2, dropout_rate: float = 0.5):
        """
        初始化CNN模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            max_length: 最大序列长度
            num_classes: 分类数量
            dropout_rate: Dropout比率
        """
        super(LightweightCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN层
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 嵌入层 [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        
        # 转换维度以适应Conv1d [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # 第一层卷积 + 批归一化 + 激活 + 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 第二层卷积 + 批归一化 + 激活 + 池化
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 第三层卷积 + 批归一化 + 激活 + 池化
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 自适应池化
        x = self.pool(x)
        
        # 展平 [batch_size, channels, 1] -> [batch_size, channels]
        x = x.squeeze(-1)
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None) -> Dict[str, Any]:
    """
    评估预测结果
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        
    Returns:
        评估结果
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 构建结果字典
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    # 如果提供了类别名称，计算每个类别的指标
    if class_names:
        # 计算每个类别的精确率、召回率和F1分数
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred)
        
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                class_metrics[class_name] = {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1_score': f1_per_class[i]
                }
        
        results['class_metrics'] = class_metrics
    
    return results


if __name__ == "__main__":
    print("轻量级CNN模型模块")
    print("使用 core.trainer.Trainer 进行训练和预测")