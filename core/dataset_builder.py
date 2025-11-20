"""
统一数据集构建器
支持多分类标签格式，可根据配置转换为二分类或多分类任务
"""

import pickle
import random
import numpy as np
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path

class DatasetBuilder:
    """统一数据集构建器"""
    
    def __init__(self, data_dir: str = "data", use_selected_families: bool = True):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.dga_dir = self.raw_dir / "DGA_Botnets_Domains"  # 恶意域名目录
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        if use_selected_families:
            # 使用精选的恶意域名家族（主要是字符级DGA + 少量字典级）
            self.malicious_families = self._get_selected_families()
        else:
            # 从文件名自动获取所有恶意域名类别
            self.malicious_families = self._get_all_families()
        
        # 标签映射
        self.label_mapping = {
            'benign': 0,  # 良性域名
            **{family: i+1 for i, family in enumerate(self.malicious_families)}
        }
        
        print(f"\n使用 {len(self.malicious_families)} 个恶意域名家族:")
        print(f"良性域名: benign (标签=0)")
        for i, family in enumerate(self.malicious_families):
            print(f"恶意家族 {i+1:2d}: {family:20s} (标签={i+1})")
    
    def _get_selected_families(self) -> List[str]:
        """获取精选的恶意域名家族列表
        
        根据Kimi对话分析结果，选择代表性的DGA家族：
        - 主要选择字符级DGA（随机字符生成）
        - 包含少量字典级DGA（单词组合）
        - 优先选择样本数>=50000的家族
        - 总类别控制在20-25个左右
        """
        # 字符级DGA家族（19个）- 这些生成随机字符域名
        character_based = [
            'banjori',        # 算术基础，随机字符
            'cryptolocker',   # 著名勒索软件，随机字符
            'ramnit',         # 知名僵尸网络，随机字符
            'tinba',          # 银行木马，随机字符
            'simda',          # 点击欺诈，随机字符
            'necurs',         # 垃圾邮件僵尸网络，随机字符
            'locky',          # 勒索软件，随机字符
            'pushdo',         # 垃圾邮件，随机字符
            'qakbot',         # 银行木马，随机字符
            'ramdo',          # 随机字符
            'virut',          # 简单6字符设计
            'emotet',         # 著名银行木马
            'dyre',           # 银行木马
            'bamital',        # MD5哈希生成
            'torpig',         # 点击欺诈
            'zloader',        # 银行木马
            'qsnatch',        # NAS恶意软件
            'sisron',         # 随机字符
            'tempedreve',     # 随机字符
        ]
        
        # 字典级DGA家族（5个）- 这些使用单词组合
        dictionary_based = [
            'matsnu',         # 使用动词和名词词典
            'suppobox_1',     # 使用384个常用英语单词
            'gozi_gpl',       # 使用GPL文档中的单词
            'nymaim2',        # 使用两个词典（2450词和4387词）
            'rovnix',         # 使用美国独立宣言
        ]
        
        # 合并：19个字符级 + 5个字典级 = 24个恶意家族 + 1个良性 = 25类
        selected = character_based + dictionary_based
        
        print(f"\n=== 数据集类别设计 ===")
        print(f"字符级DGA家族: {len(character_based)} 个")
        print(f"字典级DGA家族: {len(dictionary_based)} 个")
        print(f"总恶意家族数: {len(selected)} 个")
        print(f"总类别数（含良性）: {len(selected) + 1} 类")
        
        return selected
    
    def _get_all_families(self) -> List[str]:
        """从文件名自动获取所有恶意域名家族列表（旧方法）"""
        families = []
        
        if not self.dga_dir.exists():
            print(f"警告: 原始数据目录不存在: {self.dga_dir}")
            return self._get_selected_families()
        
        for file_path in self.dga_dir.glob("*.txt"):
            filename = file_path.stem  # 获取不带扩展名的文件名
            
            # 提取家族名称（去掉数字后缀）
            family_name = filename.split('-')[0].lower()
            
            # 跳过良性域名文件
            if family_name in ['legit', 'benign', 'legitimate']:
                continue
                
            if family_name not in families:
                families.append(family_name)
        
        return sorted(families)
    
    def _load_domains_from_file(self, file_path: Path, max_samples: int = None) -> List[str]:
        """从文件加载域名"""
        domains = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    domain = line.strip()
                    if domain and not domain.startswith('#'):  # 跳过空行和注释
                        # 清理域名（移除可能的URL前缀）
                        if '://' in domain:
                            domain = domain.split('://')[1]
                        if '/' in domain:
                            domain = domain.split('/')[0]
                        
                        domains.append(domain)
                        if max_samples and len(domains) >= max_samples:
                            break
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            
        return domains
    
    def _load_benign_domains(self, max_samples: int) -> List[str]:
        """从legit-1000000.txt加载良性域名"""
        benign_file = self.dga_dir / 'legit-1000000.txt'
        
        if not benign_file.exists():
            raise FileNotFoundError(f"良性域名文件不存在: {benign_file}")
        
        print(f"从 {benign_file.name} 加载良性域名...")
        domains = self._load_domains_from_file(benign_file, max_samples)
        
        if len(domains) < max_samples:
            print(f"警告: 只找到 {len(domains)} 个良性域名，需要 {max_samples} 个")
        else:
            print(f"✓ 成功加载 {len(domains)} 个良性域名")
        
        return domains[:max_samples]
    

    
    def build_dataset(self, size: str, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, Any]:
        """
        构建数据集
        
        Args:
            size: 数据集大小 ('small', 'medium', 'large', '500k')
            split_ratio: 训练/验证/测试集划分比例
            
        Returns:
            包含训练、验证、测试数据的字典
        """
        # 数据集大小配置
        size_config = {
            'small': 10000,
            'medium': 50000, 
            'large': 200000,
            '500k': 500000
        }
        
        total_samples = size_config[size]
        
        # 50% 良性，50% 恶意（平均分配给各恶意家族）
        benign_samples = total_samples // 2
        malicious_samples = total_samples // 2
        
        # 计算每个家族需要的样本数
        families_to_use = self.malicious_families
        samples_per_family = malicious_samples // len(families_to_use)
        
        print(f"每个恶意家族目标样本数: {samples_per_family}")
        
        print(f"构建 {size} 数据集 (总样本: {total_samples})")
        print(f"良性样本: {benign_samples}")
        if size == '500k':
            print(f"恶意样本: {malicious_samples} ({len(families_to_use)} 个家族，每个家族: {samples_per_family})")
        else:
            print(f"恶意样本: {malicious_samples} (每个家族: {samples_per_family})")
        
        # 加载数据
        domains = []
        labels = []
        
        # 加载良性域名
        print("加载良性域名...")
        benign_domains = self._load_benign_domains(benign_samples)
        for domain in benign_domains:
            domains.append(domain)
            labels.append(self.label_mapping['benign'])
        
        # 加载恶意域名
        print(f"\n加载恶意域名 (共 {len(families_to_use)} 个家族)...")
        loaded_families = 0
        failed_families = []
        
        for family in families_to_use:
            print(f"  [{loaded_families+1}/{len(families_to_use)}] 加载 {family:20s} 家族...", end=" ")
            
            # 查找对应的文件（精确匹配文件名模式）
            family_files = list(self.dga_dir.glob(f"{family}-*.txt"))
            
            # 如果精确匹配失败，尝试模糊匹配
            if not family_files:
                family_files = list(self.dga_dir.glob(f"{family}*.txt"))
            
            if not family_files:
                print(f"❌ 未找到数据文件，跳过")
                failed_families.append(family)
                continue
            
            # 从第一个匹配的文件加载域名
            family_file = family_files[0]
            family_domains = self._load_domains_from_file(family_file, samples_per_family)
            
            if len(family_domains) < samples_per_family:
                print(f"⚠️  只加载了 {len(family_domains):5d}/{samples_per_family} 个域名 ({family_file.name})")
            else:
                print(f"✓ 加载了 {len(family_domains):5d} 个域名 ({family_file.name})")
            
            for domain in family_domains:
                domains.append(domain)
                labels.append(self.label_mapping[family])
            
            loaded_families += 1
        
        if failed_families:
            print(f"\n警告: 以下家族未找到数据文件: {failed_families}")
        
        # 检查实际加载的样本数
        actual_samples = len(domains)
        print(f"实际加载样本数: {actual_samples}")
        
        if actual_samples < total_samples:
            print(f"样本不足，目标: {total_samples}, 实际: {actual_samples}")
        
        # 打乱数据
        combined = list(zip(domains, labels))
        random.shuffle(combined)
        domains, labels = zip(*combined)
        
        # 划分数据集
        train_size = int(len(domains) * split_ratio[0])
        val_size = int(len(domains) * split_ratio[1])
        
        train_domains = domains[:train_size]
        train_labels = labels[:train_size]
        
        val_domains = domains[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        
        test_domains = domains[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
        
        dataset = {
            'train': {
                'domains': list(train_domains),
                'labels': list(train_labels)
            },
            'val': {
                'domains': list(val_domains),
                'labels': list(val_labels)
            },
            'test': {
                'domains': list(test_domains),
                'labels': list(test_labels)
            },
            'metadata': {
                'total_samples': total_samples,
                'num_classes': len(self.label_mapping),
                'label_mapping': self.label_mapping,
                'malicious_families': self.malicious_families,
                'split_ratio': split_ratio,
                'size': size
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filename: str):
        """保存数据集"""
        filepath = self.processed_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"数据集已保存到: {filepath}")
    
    def build_all_datasets(self):
        """构建所有规格的数据集"""
        sizes = ['small', 'medium', 'large']
        
        for size in sizes:
            print(f"\n{'='*50}")
            print(f"构建 {size.upper()} 数据集")
            print(f"{'='*50}")
            
            dataset = self.build_dataset(size)
            filename = f"{size}_unified_dga_dataset.pkl"
            self.save_dataset(dataset, filename)
            
            # 打印统计信息
            self._print_dataset_stats(dataset)
    
    def _print_dataset_stats(self, dataset: Dict[str, Any]):
        """打印数据集统计信息"""
        metadata = dataset['metadata']
        
        print(f"\n数据集统计:")
        print(f"总样本数: {metadata['total_samples']}")
        print(f"类别数: {metadata['num_classes']}")
        print(f"划分比例: {metadata['split_ratio']}")
        
        for split in ['train', 'val', 'test']:
            labels = dataset[split]['labels']
            print(f"\n{split.upper()} 集:")
            print(f"  样本数: {len(labels)}")
            
            # 统计各类别样本数
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # 按标签排序显示
            for label in sorted(label_counts.keys()):
                if label == 0:
                    class_name = "良性"
                else:
                    family_name = metadata['malicious_families'][label-1]
                    class_name = f"恶意-{family_name}"
                print(f"    {class_name}: {label_counts[label]}")

def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建数据集构建器
    builder = DatasetBuilder()
    
    # 构建50万样本数据集
    print(f"\n{'='*50}")
    print("构建 500K 数据集")
    print(f"{'='*50}")
    
    dataset = builder.build_dataset('500k')
    filename = "500k_unified_dga_dataset.pkl"
    builder.save_dataset(dataset, filename)
    
    # 打印统计信息
    builder._print_dataset_stats(dataset)
    
    print(f"\n{'='*50}")
    print("50万样本数据集构建完成！")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()