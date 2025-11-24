#!/bin/bash
# GAN对抗样本生成与攻击实验脚本
# 用途：训练WGAN-GP生成器与CNN判别器，并对目标分类器执行攻击评估

set -e  # 遇到错误即退出

# ==================== 配置参数 ====================
EPOCHS=50
BATCH_SIZE=512
LR=1e-4
Z_DIM=100
HIDDEN_DIM=256
DISCRIMINATOR_TYPE="cnn"
DATASET_PATH="./data/processed/500k_unified_dga_dataset.pkl"

# 攻击参数
TARGET_MODEL="./models/cnn_binary_model.pth"
NUM_ATTACK_SAMPLES=1000

# ==================== 步骤1: 训练GAN ====================
echo "======================================"
echo "步骤1: 训练WGAN-GP (生成器 + 判别器)"
echo "======================================"
echo "参数: Epochs=${EPOCHS}, BatchSize=${BATCH_SIZE}, Discriminator=${DISCRIMINATOR_TYPE}"
echo ""

python gan/train_gan.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --z_dim ${Z_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --discriminator ${DISCRIMINATOR_TYPE} \
    --dataset_path ${DATASET_PATH}

echo ""
echo "✅ GAN训练完成！"
echo ""

# ==================== 步骤2: 检查生成的模型 ====================
echo "======================================"
echo "步骤2: 检查生成器模型文件"
echo "======================================"
ls -lh ./models/gan/generator_epoch_*.pth 2>/dev/null || echo "⚠️ 未找到生成器权重文件"
echo ""

# ==================== 步骤3: 对抗攻击评估 ====================
echo "======================================"
echo "步骤3: 使用生成器对目标分类器执行攻击"
echo "======================================"
echo "目标模型: ${TARGET_MODEL}"
echo "生成样本数: ${NUM_ATTACK_SAMPLES}"
echo ""

# 选择最后一个epoch的生成器
GENERATOR_PATH="./models/gan/generator_epoch_${EPOCHS}.pth"

if [ -f "${GENERATOR_PATH}" ]; then
    python gan/attack_model.py \
        --target_model_path ${TARGET_MODEL} \
        --generator_path ${GENERATOR_PATH} \
        --dataset_path ${DATASET_PATH} \
        --num_samples ${NUM_ATTACK_SAMPLES} \
        --z_dim ${Z_DIM} \
        --hidden_dim ${HIDDEN_DIM}
    
    echo ""
    echo "✅ 攻击评估完成！"
else
    echo "❌ 未找到生成器文件: ${GENERATOR_PATH}"
    exit 1
fi

# ==================== 步骤4: GPU使用情况 ====================
echo ""
echo "======================================"
echo "步骤4: GPU使用情况"
echo "======================================"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "🎉 实验流程全部完成！"
