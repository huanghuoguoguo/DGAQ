#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用训练器框架 - 模型无关的训练抽象
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import time
import logging
import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import TrainingLogger


class Trainer:
    """通用训练器类 - 支持任意PyTorch模型的训练"""
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            criterion: 损失函数
            optimizer: 优化器
            device: 训练设备
            scheduler: 学习率调度器（可选）
            logger: 日志记录器（可选）
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger(__name__)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _is_moe_model(self) -> bool:
        """检测模型是否是MoE模型（有compute_loss方法）"""
        return hasattr(self.model, 'compute_loss') and callable(getattr(self.model, 'compute_loss'))
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int, num_epochs: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch编号（从0开始）
            num_epochs: 总epoch数
            
        Returns:
            包含损失和准确率的字典
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # MoE模型的额外统计
        is_moe = self._is_moe_model()
        expert_stats = None
        if is_moe:
            num_experts = getattr(self.model, 'num_experts', 0)
            num_layers = getattr(self.model, 'num_layers', len(self.model.layers)) if hasattr(self.model, 'layers') else 1
            if num_experts > 0:
                expert_stats = torch.zeros(num_layers, num_experts).to(self.device)
        
        num_batches = len(train_loader)
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"开始第 {epoch+1}/{num_epochs} 轮训练")
        print(f"{'='*50}")
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # 将数据移动到指定设备
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播 - 根据模型类型选择不同的调用方式
            if is_moe:
                # MoE模型：返回logits和门控信息
                outputs, gate_info = self.model(data, return_gate=True)
                loss = self.model.compute_loss(outputs, targets, gate_info)
                
                # 统计专家使用情况
                if expert_stats is not None and 'expert_usage' in gate_info:
                    expert_stats += gate_info['expert_usage']
            else:
                # 标准模型
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 统计损失和准确率
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 打印进度 - 每10%显示一次
            if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                avg_loss = train_loss / (batch_idx + 1)
                acc = 100. * train_correct / train_total
                elapsed_time = time.time() - start_time
                
                # 显示进度条
                bar_length = 30
                filled_length = int(bar_length * (batch_idx + 1) // num_batches)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f'\rEpoch [{epoch+1}/{num_epochs}] |{bar}| {progress:.1f}% '
                      f'Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, '
                      f'Acc: {acc:.2f}%, Time: {elapsed_time:.1f}s', end='')
        
        print()  # 换行
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / num_batches
        train_acc = 100. * train_correct / train_total
        
        result = {
            'loss': avg_train_loss,
            'accuracy': train_acc
        }
        
        # 如果是MoE模型，添加专家使用统计
        if is_moe and expert_stats is not None:
            # expert_stats: [num_layers, num_experts]
            avg_expert_usage = (expert_stats / num_batches).cpu()  # [num_layers, num_experts]
            result['expert_usage'] = avg_expert_usage.tolist()  # 保存为列表的列表
        
        return result
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            包含损失和准确率的字典
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # MoE模型的额外统计
        is_moe = self._is_moe_model()
        expert_stats = None
        if is_moe:
            num_experts = getattr(self.model, 'num_experts', 0)
            num_layers = getattr(self.model, 'num_layers', len(self.model.layers)) if hasattr(self.model, 'layers') else 1
            if num_experts > 0:
                expert_stats = torch.zeros(num_layers, num_experts).to(self.device)
        
        print(f"开始验证...")
        
        with torch.no_grad():
            val_batches = len(val_loader)
            for batch_idx, (data, targets) in enumerate(val_loader):
                # 将数据移动到指定设备
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 前向传播 - 根据模型类型选择不同的调用方式
                if is_moe:
                    # MoE模型：返回logits和门控信息
                    outputs, gate_info = self.model(data, return_gate=True)
                    loss = self.model.compute_loss(outputs, targets, gate_info)
                    
                    # 统计专家使用情况
                    if expert_stats is not None and 'expert_usage' in gate_info:
                        expert_stats += gate_info['expert_usage']
                else:
                    # 标准模型
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                # 统计损失和准确率
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 显示验证进度
                if batch_idx % max(1, val_batches // 5) == 0 or batch_idx == val_batches - 1:
                    progress = (batch_idx + 1) / val_batches * 100
                    bar_length = 20
                    filled_length = int(bar_length * (batch_idx + 1) // val_batches)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f'\r验证进度 |{bar}| {progress:.1f}%', end='')
        
        print()  # 换行
        
        # 计算平均损失和准确率
        avg_val_loss = val_loss / val_batches
        val_acc = 100. * val_correct / val_total
        
        result = {
            'loss': avg_val_loss,
            'accuracy': val_acc
        }
        
        # 如果是MoE模型，添加专家使用统计
        if is_moe and expert_stats is not None:
            # expert_stats: [num_layers, num_experts]
            avg_expert_usage = (expert_stats / val_batches).cpu()  # [num_layers, num_experts]
            result['expert_usage'] = avg_expert_usage.tolist()  # 保存为列表的列表
        
        return result
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              num_epochs: int,
              callbacks: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            callbacks: 回调函数字典（可选）
                - on_epoch_end: 每个epoch结束时调用
                - on_train_end: 训练结束时调用
                
        Returns:
            训练历史记录
        """
        self.logger.info(f"开始训练，共 {num_epochs} 轮")
        self.logger.info(f"使用设备: {self.device}")
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            
            # 验证阶段
            val_metrics = self.validate_epoch(val_loader)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # 打印轮次结果
            print(f'Epoch [{epoch+1}/{num_epochs}] 结果:')
            print(f'  训练 - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'  验证 - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2f}%')
            print(f'  学习率: {current_lr:.6f}')
            
            # 如果是MoE模型，打印专家使用统计
            if 'expert_usage' in train_metrics:
                expert_usage = train_metrics['expert_usage']  # [[layer1_exp1, layer1_exp2, ...], [layer2_exp1, ...]]
                num_layers = len(expert_usage)
                num_experts = len(expert_usage[0]) if num_layers > 0 else 0
                
                print(f'\n  📊 MoE专家使用统计（训练集）:')
                
                # 打印每层的专家使用情况
                for layer_idx in range(num_layers):
                    print(f'\n    第 {layer_idx+1} 层:')
                    for expert_idx in range(num_experts):
                        usage = expert_usage[layer_idx][expert_idx]
                        bar_len = int(usage * 50)
                        bar = '█' * bar_len + '░' * (50 - bar_len)
                        print(f'      专家 {expert_idx+1}: [{bar}] {usage*100:.2f}%')
                
                # 检测专家塌陷（平均所有层）
                avg_usage_per_expert = [sum(expert_usage[l][e] for l in range(num_layers)) / num_layers 
                                       for e in range(num_experts)]
                expected = 1.0 / num_experts
                max_imbalance = max([abs(u - expected) for u in avg_usage_per_expert])
                
                if max_imbalance > 0.2:  # 如果偏差超过20%
                    print(f'\n     ⚠️  警告: 专家负载不均衡! 最大偏差={max_imbalance*100:.1f}%')
                if any(u < 0.05 for u in avg_usage_per_expert):  # 如果有专家使用率<5%
                    print(f'     ⚠️  警告: 检测到专家塌陷! 某些专家几乎不被使用')
            
            if 'expert_usage' in val_metrics:
                expert_usage = val_metrics['expert_usage']
                num_layers = len(expert_usage)
                num_experts = len(expert_usage[0]) if num_layers > 0 else 0
                
                print(f'\n  📊 MoE专家使用统计（验证集）:')
                
                for layer_idx in range(num_layers):
                    print(f'\n    第 {layer_idx+1} 层:')
                    for expert_idx in range(num_experts):
                        usage = expert_usage[layer_idx][expert_idx]
                        bar_len = int(usage * 50)
                        bar = '█' * bar_len + '░' * (50 - bar_len)
                        print(f'      专家 {expert_idx+1}: [{bar}] {usage*100:.2f}%')
            
            log_msg = (f'Epoch [{epoch+1}/{num_epochs}] - '
                      f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}%, '
                      f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
            if 'expert_usage' in train_metrics:
                # 记录专家使用详情（平均所有层）
                expert_usage = train_metrics['expert_usage']
                num_layers = len(expert_usage)
                num_experts = len(expert_usage[0]) if num_layers > 0 else 0
                avg_usage_per_expert = [sum(expert_usage[l][e] for l in range(num_layers)) / num_layers 
                                       for e in range(num_experts)]
                expert_usage_str = ', '.join([f'E{i+1}:{u*100:.1f}%' for i, u in enumerate(avg_usage_per_expert)])
                log_msg += f', Expert Usage: [{expert_usage_str}]'
            self.logger.info(log_msg)
            
            # 执行回调函数
            if callbacks and 'on_epoch_end' in callbacks:
                callbacks['on_epoch_end'](epoch, train_metrics, val_metrics)
        
        # 执行训练结束回调
        if callbacks and 'on_train_end' in callbacks:
            callbacks['on_train_end'](self.history)
        
        self.logger.info("模型训练完成")
        print("\n模型训练完成!")
        
        return self.history
    
    def predict(self, test_loader: torch.utils.data.DataLoader):
        """
        在测试集上进行预测
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            预测结果和真实标签
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                # 将数据移动到指定设备
                data = data.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 获取预测结果
                _, predicted = outputs.max(1)
                
                # 收集结果
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(targets.numpy())
        
        import numpy as np
        return np.array(predictions), np.array(true_labels)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None) -> Dict[str, Any]:
        """
        评估预测结果
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            
        Returns:
            评估结果字典
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
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"模型已保存至: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
        self.logger.info(f"模型已从 {path} 加载")


class TrainerBuilder:
    """训练器构建器 - 用于简化训练器的创建"""
    
    def __init__(self, model: nn.Module, model_name: str = 'model', task_type: str = 'task'):
        """
        初始化训练器构建器
        
        Args:
            model: PyTorch模型
            model_name: 模型名称，用于日志文件命名
            task_type: 任务类型，用于日志文件命名
        """
        self.model = model
        self.model_name = model_name
        self.task_type = task_type
        self._criterion = None
        self._optimizer = None
        self._scheduler = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger = None
        self._use_custom_logger = True
    
    def with_criterion(self, criterion: nn.Module):
        """设置损失函数"""
        self._criterion = criterion
        return self
    
    def with_optimizer(self, optimizer_class, **kwargs):
        """设置优化器"""
        self._optimizer = optimizer_class(self.model.parameters(), **kwargs)
        return self
    
    def with_scheduler(self, scheduler_class, **kwargs):
        """设置学习率调度器"""
        if self._optimizer is None:
            raise ValueError("必须先设置优化器才能设置调度器")
        self._scheduler = scheduler_class(self._optimizer, **kwargs)
        return self
    
    def with_device(self, device: torch.device):
        """设置训练设备"""
        self._device = device
        return self
    
    def with_logger(self, logger: logging.Logger):
        """设置日志记录器（外部提供）"""
        self._logger = logger
        self._use_custom_logger = False
        return self
    
    def with_auto_logger(self, log_dir: str = './logs'):
        """使用自动创建的训练日志记录器"""
        logger_manager = TrainingLogger(log_dir=log_dir)
        self._logger = logger_manager.setup_logger(self.model_name, self.task_type)
        self._use_custom_logger = True
        return self
    
    def build(self) -> Trainer:
        """构建训练器"""
        if self._criterion is None:
            raise ValueError("必须设置损失函数")
        if self._optimizer is None:
            raise ValueError("必须设置优化器")
        
        # 如果没有设置logger，自动创建一个
        if self._logger is None and self._use_custom_logger:
            logger_manager = TrainingLogger()
            self._logger = logger_manager.setup_logger(self.model_name, self.task_type)
        
        return Trainer(
            model=self.model,
            criterion=self._criterion,
            optimizer=self._optimizer,
            device=self._device,
            scheduler=self._scheduler,
            logger=self._logger
        )


if __name__ == "__main__":
    print("通用训练器框架模块")
    print("使用示例:")
    print("""
    # 创建模型
    model = YourModel()
    
    # 使用构建器创建训练器
    trainer = TrainerBuilder(model) \\
        .with_criterion(nn.CrossEntropyLoss()) \\
        .with_optimizer(torch.optim.Adam, lr=0.001) \\
        .with_scheduler(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5) \\
        .with_device(torch.device('cuda')) \\
        .build()
    
    # 开始训练
    history = trainer.train(train_loader, val_loader, num_epochs=10)
    
    # 预测
    predictions, labels = trainer.predict(test_loader)
    
    # 保存模型
    trainer.save_model('model.pth')
    """)