#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志管理器
支持按时间-模型-任务类型格式命名日志文件
"""

import logging
import os
from datetime import datetime
from typing import Optional


class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, log_dir: str = "./logs"):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 清理旧的日志配置
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
    def setup_logger(self, model_name: str, task_type: str, 
                    level: int = logging.INFO) -> logging.Logger:
        """
        设置日志记录器
        
        Args:
            model_name: 模型名称 (如 lstm, cnn)
            task_type: 任务类型 (如 binary, multiclass)
            level: 日志级别
            
        Returns:
            配置好的日志记录器
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成日志文件名: 时间-模型-任务类型.log
        log_filename = f"{timestamp}-{model_name}-{task_type}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        # 创建日志记录器
        logger = logging.getLogger(f"training_{model_name}_{task_type}")
        logger.setLevel(level)
        
        # 清除已有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器 - 使用UTF-8编码
        file_handler = logging.FileHandler(
            log_path, 
            mode='w', 
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        # 控制台处理器 - 使用UTF-8编码
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 设置格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # 防止重复输出
        logger.propagate = False
        
        # 记录日志文件路径
        logger.info(f"日志文件: {log_path}")
        
        return logger
    
    def get_latest_log_file(self, model_name: Optional[str] = None, 
                           task_type: Optional[str] = None) -> Optional[str]:
        """
        获取最新的日志文件路径
        
        Args:
            model_name: 模型名称过滤
            task_type: 任务类型过滤
            
        Returns:
            最新日志文件路径，如果没有找到返回None
        """
        if not os.path.exists(self.log_dir):
            return None
        
        log_files = []
        for filename in os.listdir(self.log_dir):
            if not filename.endswith('.log'):
                continue
                
            # 检查文件名格式: YYYYMMDD_HHMMSS-model-task.log
            parts = filename.replace('.log', '').split('-')
            if len(parts) != 3:
                continue
                
            timestamp_part, file_model, file_task = parts
            
            # 应用过滤条件
            if model_name and file_model != model_name:
                continue
            if task_type and file_task != task_type:
                continue
                
            log_files.append((timestamp_part, filename))
        
        if not log_files:
            return None
            
        # 按时间戳排序，返回最新的
        log_files.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(self.log_dir, log_files[0][1])
    
    def list_log_files(self) -> list:
        """
        列出所有日志文件
        
        Returns:
            日志文件信息列表 [(时间戳, 模型, 任务类型, 文件路径)]
        """
        if not os.path.exists(self.log_dir):
            return []
        
        log_files = []
        for filename in os.listdir(self.log_dir):
            if not filename.endswith('.log'):
                continue
                
            # 检查文件名格式
            parts = filename.replace('.log', '').split('-')
            if len(parts) != 3:
                continue
                
            timestamp_part, model, task = parts
            file_path = os.path.join(self.log_dir, filename)
            
            log_files.append((timestamp_part, model, task, file_path))
        
        # 按时间戳排序
        log_files.sort(key=lambda x: x[0], reverse=True)
        return log_files
    
    def clean_old_logs(self, keep_count: int = 10):
        """
        清理旧的日志文件，保留最新的几个
        
        Args:
            keep_count: 保留的日志文件数量
        """
        log_files = self.list_log_files()
        
        if len(log_files) <= keep_count:
            return
        
        # 删除多余的旧日志文件
        for _, _, _, file_path in log_files[keep_count:]:
            try:
                os.remove(file_path)
                print(f"已删除旧日志文件: {file_path}")
            except Exception as e:
                print(f"删除日志文件失败 {file_path}: {e}")


def get_training_logger(model_name: str, task_type: str) -> logging.Logger:
    """
    便捷函数：获取训练日志记录器
    
    Args:
        model_name: 模型名称
        task_type: 任务类型
        
    Returns:
        配置好的日志记录器
    """
    logger_manager = TrainingLogger()
    return logger_manager.setup_logger(model_name, task_type)


if __name__ == "__main__":
    # 测试日志系统
    logger_manager = TrainingLogger()
    
    # 创建测试日志
    logger = logger_manager.setup_logger("lstm", "binary")
    logger.info("这是一个测试日志消息")
    logger.info("测试中文字符显示")
    logger.warning("这是警告消息")
    logger.error("这是错误消息")
    
    # 列出日志文件
    print("\n当前日志文件:")
    for timestamp, model, task, path in logger_manager.list_log_files():
        print(f"  {timestamp} - {model} - {task}: {path}")