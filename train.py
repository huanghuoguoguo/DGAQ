#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练框架 - 配置驱动的多模型训练和比对
支持从配置文件读取参数，按顺序训练多个模型，最后比对结果
"""

import torch
import argparse
import os
import sys
from typing import Dict, Any, List
import time
from datetime import datetime
import pandas as pd

# 使用Python 3.11+自带的tomllib，或使用toml库
try:
    import tomllib
except ImportError:
    import toml as tomllib

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.logger import get_training_logger
from core.trainer import TrainerBuilder


class ModelFactory:
    """模型工厂 - 根据配置创建不同类型的模型"""
    
    @staticmethod
    def create_model(model_name: str, model_params: Dict, dataset_info: Dict):
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            model_params: 模型参数
            dataset_info: 数据集信息
        """
        vocab_size = dataset_info['vocab_size']
        max_length = dataset_info['max_length']
        num_classes = dataset_info['num_classes']
        
        if model_name == 'cnn':
            from core.model.cnn_model import LightweightCNN
            return LightweightCNN(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 128),
                max_length=max_length,
                num_classes=num_classes,
                dropout_rate=model_params.get('dropout_rate', 0.5)
            )
        
        elif model_name == 'transformer':
            from core.model.transformer_model import LightweightTransformer
            return LightweightTransformer(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 128),
                max_length=max_length,
                num_classes=num_classes,
                num_heads=model_params.get('num_heads', 4),
                num_layers=model_params.get('num_layers', 2),
                dim_feedforward=model_params.get('dim_feedforward', 256),
                dropout_rate=model_params.get('dropout_rate', 0.3)
            )
        
        elif model_name == 'mamba':
            from core.model.mamba1_model import LightweightMamba
            return LightweightMamba(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 128),
                max_length=max_length,
                num_classes=num_classes,
                num_layers=model_params.get('num_layers', 2),
                d_state=model_params.get('d_state', 16),
                d_conv=model_params.get('d_conv', 4),
                expand=model_params.get('expand', 2),
                dropout_rate=model_params.get('dropout_rate', 0.3)
            )
        
        elif model_name == 'mamba2':
            from core.model.mamba2_model import LightweightMamba2
            return LightweightMamba2(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 256),  # Mamba2需要更大的embedding_dim
                max_length=max_length,
                num_classes=num_classes,
                num_layers=model_params.get('num_layers', 2),
                d_state=model_params.get('d_state', 128),
                d_conv=model_params.get('d_conv', 4),
                expand=model_params.get('expand', 2),
                headdim=model_params.get('headdim', 64),
                dropout_rate=model_params.get('dropout_rate', 0.3)
            )
        
        elif model_name == 'cnn_moe':
            from core.model.cnn_moe_model import LightweightCNNMoE
            return LightweightCNNMoE(
                vocab_size=vocab_size,
                num_classes=num_classes,
                num_experts=model_params.get('num_experts', 3),
                aux_weight=model_params.get('aux_weight', 0.3),
                balance_weight=model_params.get('balance_weight', 0.01),
                dropout_rate=model_params.get('dropout_rate', 0.5)
            )
        
        elif model_name == 'transformer_moe':
            from core.model.transformer_moe_model import LightweightTransformerMoE
            return LightweightTransformerMoE(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 128),
                max_length=max_length,
                num_classes=num_classes,
                num_heads=model_params.get('num_heads', 4),
                num_layers=model_params.get('num_layers', 2),
                num_experts=model_params.get('num_experts', 3),
                aux_weight=model_params.get('aux_weight', 0.3),
                balance_weight=model_params.get('balance_weight', 0.01),
                dropout_rate=model_params.get('dropout_rate', 0.3)
            )
        
        elif model_name == 'mamba_moe':
            from core.model.mamba1_moe_model import LightweightMambaMoE
            return LightweightMambaMoE(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 128),
                max_length=max_length,
                num_classes=num_classes,
                num_layers=model_params.get('num_layers', 2),
                d_state=model_params.get('d_state', 16),
                num_experts=model_params.get('num_experts', 3),
                aux_weight=model_params.get('aux_weight', 0.3),
                balance_weight=model_params.get('balance_weight', 0.01),
                dropout_rate=model_params.get('dropout_rate', 0.3)
            )
        
        elif model_name == 'tcbam':
            from core.model.tcbam_model import LightweightTCBAM
            return LightweightTCBAM(
                vocab_size=vocab_size,
                embedding_dim=model_params.get('embedding_dim', 64),
                num_heads=model_params.get('num_heads', 4),
                num_classes=num_classes,
                max_length=max_length,
                dropout_rate=model_params.get('dropout_rate', 0.3)
            )
        
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")


def train_single_model(model_config: Dict, global_config: Dict, 
                       train_loader, val_loader, test_loader, 
                       dataset_info: Dict, device) -> Dict[str, Any]:
    """
    训练单个模型
    
    Args:
        model_config: 模型配置
        global_config: 全局配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        dataset_info: 数据集信息
        device: 设备
    
    Returns:
        包含训练结果的字典
    """
    model_name = model_config['name']
    task_type = global_config['task_type']
    
    # 创建日志记录器
    logger = get_training_logger(model_name, task_type)
    logger.info(f"{'='*60}")
    logger.info(f"开始训练模型: {model_name}")
    logger.info(f"任务类型: {'二分类' if task_type == 'binary' else '多分类'}")
    logger.info(f"模型参数: {model_config.get('model_params', {})}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 创建模型
        logger.info("正在创建模型...")
        model = ModelFactory.create_model(
            model_name=model_name,
            model_params=model_config.get('model_params', {}),
            dataset_info=dataset_info
        )
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数量: {total_params:,}")
        
        # 创建训练器
        logger.info("正在创建训练器...")
        trainer = TrainerBuilder(model, model_name=model_name, task_type=task_type) \
            .with_criterion(torch.nn.CrossEntropyLoss()) \
            .with_optimizer(torch.optim.Adam, lr=global_config['learning_rate']) \
            .with_scheduler(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5) \
            .with_device(device) \
            .with_logger(logger) \
            .build()
        
        # 训练模型
        logger.info(f"开始训练 (共 {global_config['epochs']} 轮)...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=global_config['epochs']
        )
        
        # 在测试集上预测
        logger.info("正在测试集上进行预测...")
        y_pred, y_true = trainer.predict(test_loader)
        
        # 评估结果
        logger.info("正在评估预测结果...")
        eval_results = trainer.evaluate(y_true, y_pred, dataset_info['class_names'])
        
        # 打印评估结果
        logger.info("=" * 60)
        logger.info("模型评估结果:")
        logger.info(f"准确率: {eval_results['accuracy']:.4f}")
        logger.info(f"精确率: {eval_results['precision']:.4f}")
        logger.info(f"召回率: {eval_results['recall']:.4f}")
        logger.info(f"F1分数: {eval_results['f1_score']:.4f}")
        
        # 打印每个类别的详细指标
        if 'class_metrics' in eval_results and eval_results['class_metrics']:
            logger.info("\n各类别详细指标:")
            logger.info(f"{'类别':<20} {'精确率':>10} {'召回率':>10} {'F1分数':>10}")
            logger.info("-" * 60)
            for class_name, metrics in eval_results['class_metrics'].items():
                logger.info(
                    f"{class_name:<20} "
                    f"{metrics['precision']:>10.4f} "
                    f"{metrics['recall']:>10.4f} "
                    f"{metrics['f1_score']:>10.4f}"
                )
        
        logger.info("=" * 60)
        
        # 保存模型
        model_path = f"./models/{model_name}_{task_type}_model.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} 训练完成! 用时: {training_time:.2f}秒")
        
        return {
            'model_name': model_name,
            'accuracy': eval_results['accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1_score': eval_results['f1_score'],
            'params': total_params,
            'training_time': training_time,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"{model_name} 训练失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'model_name': model_name,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'params': 0,
            'training_time': time.time() - start_time,
            'status': 'failed',
            'error': str(e)
        }


def main(config_path: str = './train_config.toml'):
    """
    主函数 - 按配置文件顺序训练多个模型并比对结果
    
    Args:
        config_path: 配置文件路径
    """
    print(f"\n{'='*60}")
    print(f"统一训练框架 - 配置驱动的多模型训练")
    print(f"{'='*60}\n")
    
    # 加载配置文件
    print(f"正在加载配置文件: {config_path}")
    if hasattr(tomllib, '__spec__') and tomllib.__name__ == 'tomllib':
        # Python 3.11+ 使用 tomllib (二进制模式)
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    else:
        # 旧版本使用 toml 库 (文本模式)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = tomllib.load(f)
    
    global_config = config['global']
    models_config = config['models']
    
    # 筛选启用的模型
    enabled_models = [m for m in models_config if m.get('enabled', True)]
    print(f"启用的模型数量: {len(enabled_models)}")
    print(f"模型列表: {', '.join([m['name'] for m in enabled_models])}\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载数据集（所有模型共享）
    print("正在加载数据集...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path=global_config['dataset_path'],
        batch_size=global_config['batch_size'],
        task_type=global_config['task_type']
    )
    print_dataset_info(dataset_info)
    print()
    
    # 按顺序训练每个模型
    results = []
    for idx, model_config in enumerate(enabled_models, 1):
        print(f"\n{'#'*60}")
        print(f"训练进度: [{idx}/{len(enabled_models)}] - {model_config['name']}")
        print(f"{'#'*60}\n")
        
        result = train_single_model(
            model_config=model_config,
            global_config=global_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_info=dataset_info,
            device=device
        )
        results.append(result)
        
        print(f"\n{model_config['name']} 训练结果:")
        print(f"  状态: {result['status']}")
        if result['status'] == 'success':
            print(f"  准确率: {result['accuracy']:.4f}")
            print(f"  F1分数: {result['f1_score']:.4f}")
            print(f"  用时: {result['training_time']:.2f}秒")
        else:
            print(f"  错误: {result.get('error', 'Unknown')}")
    
    # 生成对比报告
    print(f"\n\n{'='*60}")
    print(f"所有模型训练完成! 结果对比:")
    print(f"{'='*60}\n")
    
    # 创建对比表格
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=False)
    
    # 打印对比表格
    print(df.to_string(index=False))
    
    # 保存结果到CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"./results/training_comparison_{timestamp}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存到: {results_path}")
    
    # 打印最佳模型
    best_model = df.iloc[0]
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_model['model_name']}")
    print(f"  准确率: {best_model['accuracy']:.4f}")
    print(f"  精确率: {best_model['precision']:.4f}")
    print(f"  召回率: {best_model['recall']:.4f}")
    print(f"  F1分数: {best_model['f1_score']:.4f}")
    print(f"  参数量: {best_model['params']:,}")
    print(f"  训练用时: {best_model['training_time']:.2f}秒")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='统一训练框架 - 配置驱动的多模型训练')
    parser.add_argument('--config', type=str, default='./train_config.toml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    main(config_path=args.config)
