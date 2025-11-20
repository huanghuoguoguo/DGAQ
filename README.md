# DGA Detection Project (DGAQ)

DGA（Domain Generation Algorithm，域名生成算法）检测项目，使用深度学习模型进行恶意域名识别。

## 环境配置

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04 或更高版本)
- **Python 版本**: 3.10.19
- **CUDA**: 支持 CUDA 11.8/12.4
- **GPU**: 需要 NVIDIA GPU (推荐显存 ≥ 8GB)

### Conda 环境

**环境名称**: `DGAenv`

#### 方法1: 使用 environment.yml 创建环境（推荐）

```bash
# 从 environment.yml 创建完整环境
conda env create -f environment.yml

# 激活环境
source activate DGAenv
```

#### 方法2: 手动创建环境

```bash
# 创建 Python 3.10 环境
conda create -n DGAenv python=3.10

# 激活环境
eval "$(conda shell.bash hook)" && conda activate DGAenv

# 安装PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖（使用国内镜像加速）
pip install -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn pandas

# 如需Mamba模型，安装causal-conv1d和mamba-ssm
pip install causal-conv1d==1.5.0.post8 --no-build-isolation
pip install mamba-ssm==2.2.4
```

### 核心依赖

| 库名 | 版本 | 用途 |
|------|------|------|
| torch | 2.5.1+cu124 | 深度学习框架 |
| numpy | 2.2.6 | 数值计算 |
| pandas | 2.3.3 | 数据处理 |
| scikit-learn | 1.5.2 | 机器学习工具 |
| mamba-ssm | 2.2.4 | Mamba状态空间模型 |

### 完整依赖列表

所有依赖已导出到以下文件：
- `environment.yml` - Conda 完整环境配置
- `requirements_conda.txt` - Conda 包列表
- `requirements_pip.txt` - Pip 包列表（推荐使用）

## 项目结构

```
DGAQ/
├── core/                          # 核心代码
│   ├── model/                     # 模型定义
│   │   ├── cnn_model.py          # CNN模型
│   │   ├── transformer_model.py   # Transformer模型
│   │   ├── mamba_model.py        # Mamba模型
│   │   ├── mamba2_model.py       # Mamba2模型
│   │   ├── tcbam_model.py        # TCBAM模型
│   │   ├── *_moe_model.py        # MoE变体模型
│   │   └── ...
│   ├── dataset.py                # 数据集加载
│   ├── dataset_builder.py        # 数据集构建
│   ├── logger.py                 # 日志系统
│   └── trainer.py                # 训练器框架
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   └── processed/                # 处理后的数据
├── models/                       # 保存的模型
├── logs/                         # 训练日志
├── train_*.py                    # 训练脚本
├── environment.yml               # Conda环境配置
├── requirements_pip.txt          # Pip依赖列表
└── README.md                     # 本文件
```

## 数据集

### 500k统一数据集

- **总样本数**: 499,984
- **类别数**: 25 (1个良性 + 24个恶意DGA家族)
- **划分比例**: 训练集80% / 验证集10% / 测试集10%
- **数据来源**: 良性域名（legit）+ 24个DGA家族

#### DGA家族分类

**字符级DGA (19个)**：生成随机字符域名
- banjori, cryptolocker, ramnit, tinba, simda, necurs, locky, pushdo, qakbot, ramdo, virut, emotet, dyre, bamital, torpig, zloader, qsnatch, sisron, tempedreve

**字典级DGA (5个)**：使用单词组合域名
- matsnu, suppobox_1, gozi_gpl, nymaim2, rovnix

### 构建新数据集

```bash
# 构建500k数据集
python core/dataset_builder.py
```

## 模型训练

### 支持的模型

| 模型 | 参数量 | 特点 |
|------|--------|------|
| CNN | ~89K | 轻量级，适合短序列 |
| Transformer | ~286K | 自注意力机制 |
| Mamba | ~420K | 状态空间模型 |
| Mamba2 | ~430K | 改进版Mamba |
| TCBAM | ~991K | Transformer+CBAM+DPCNN多特征融合 |
| *-MoE | 变化 | 混合专家模型变体 |

### 训练命令

#### 二分类任务（良性 vs 恶意）

```bash
# CNN模型
python train_cnn.py --task binary --epochs 10 --batch_size 32

# Transformer模型
python train_transformer.py --task binary --epochs 10 --batch_size 32

# Mamba模型
python train_mamba.py --task binary --epochs 10 --batch_size 32

# TCBAM模型
python train_tcbam.py --task binary --epochs 10 --batch_size 64
```

#### 多分类任务（25类）

```bash
# CNN模型
python train_cnn.py --task multiclass --epochs 10 --batch_size 32

# Transformer模型
python train_transformer.py --task multiclass --epochs 10 --batch_size 32

# Mamba模型
python train_mamba.py --task multiclass --epochs 10 --batch_size 32

# TCBAM模型
python train_tcbam.py --task multiclass --epochs 10 --batch_size 64
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --task | binary | 任务类型: binary/multiclass |
| --dataset | 500k数据集路径 | 数据集文件路径 |
| --batch_size | 32 | 批次大小 |
| --epochs | 10 | 训练轮数 |
| --lr | 0.001 | 学习率 |

## 环境激活

### 手动激活

```bash
# 激活conda环境
eval "$(conda shell.bash hook)" && conda activate DGAenv

# 验证环境
python --version  # 应显示 Python 3.10.19
python -c "import torch; print(torch.__version__)"  # 应显示 2.5.1+cu124
```

### 自动激活（推荐）

已配置bashrc自动激活功能，进入DGAQ目录时自动激活DGAenv环境：

```bash
cd /jsj_ywj/yhh/DGAQ  # 自动激活DGAenv环境
```

## 注意事项

1. **Mamba模型要求**: 
   - 必须在CUDA环境下运行
   - 不支持CPU模式
   - 需要先安装PyTorch，再安装 `causal-conv1d` 和 `mamba-ssm`
   - `causal-conv1d` 需要从源码编译，耗时较长（10-30分钟）

2. **环境激活**: 
   - 推荐使用 `eval "$(conda shell.bash hook)" && conda activate DGAenv`
   - 或直接进入DGAQ目录自动激活

3. **依赖安装**:
   - 推荐使用国内镜像加速：`pip install -i https://mirrors.aliyun.com/pypi/simple/ <package>`
   - PyTorch需从官方源安装以确保CUDA版本匹配
   - CUDA扩展包（causal-conv1d, mamba-ssm）必须在安装PyTorch后再安装

4. **数据集**:
   - 良性域名从 `legit-1000000.txt` 获取
   - 确保 `data/raw/DGA_Botnets_Domains/` 目录下有所需的DGA家族数据文件

5. **GPU内存**:
   - 推荐显存 ≥ 8GB
   - 可根据显存大小调整 batch_size

## 性能参考

### 二分类任务

| 模型 | 准确率 | 参数量 | 备注 |
|------|--------|--------|------|
| CNN | ~98.65% | 89K | 最优性价比 |
| Transformer | ~93.13% | 286K | 短序列表现一般 |
| Mamba-MoE | ~98.51% | 变化 | 高性能 |

### 多分类任务（25类）

| 模型 | 准确率 | 参数量 | 备注 |
|------|--------|--------|------|
| CNN | ~89-90% | 89K | 平衡性能 |
| TCBAM | 测试中 | 997K | 多特征融合 |

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请查看训练日志文件（`logs/` 目录）进行调试。
