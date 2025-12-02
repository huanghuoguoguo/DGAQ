#!/bin/bash
# 对抗训练完整流程：生成对抗样本 → 对抗训练 → 评估ASR变化

set -e  # 遇到错误立即退出

echo "=========================================="
echo "对抗训练完整流程"
echo "=========================================="
echo ""

# 配置参数
GENERATOR_PATH="./models/gan/generator_epoch_5.pth"
DATASET_PATH="./data/processed/500k_unified_dga_dataset.pkl"
ADV_SAMPLES_PATH="./data/processed/adversarial_samples.pkl"
NUM_ADV_SAMPLES=50000
ADV_RATIO=0.25

# CNN模型
CNN_PRETRAINED="./models/cnn_binary_model.pth"
CNN_ADV_OUTPUT="./models/cnn_adversarial_trained.pth"

# CNN-MoE模型
CNN_MOE_PRETRAINED="./models/cnn_moe_binary_model.pth"
CNN_MOE_ADV_OUTPUT="./models/cnn_moe_adversarial_trained.pth"

echo "Step 1: 生成对抗样本"
echo "----------------------------------------"
python gan/generate_adversarial_samples.py \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --output_path ${ADV_SAMPLES_PATH} \
    --num_samples ${NUM_ADV_SAMPLES} \
    --batch_size 1024

echo ""
echo "Step 2: CNN对抗训练"
echo "----------------------------------------"
python gan/adversarial_training.py \
    --dataset_path ${DATASET_PATH} \
    --adversarial_path ${ADV_SAMPLES_PATH} \
    --model_type cnn \
    --pretrained_path ${CNN_PRETRAINED} \
    --output_path ${CNN_ADV_OUTPUT} \
    --adv_ratio ${ADV_RATIO} \
    --epochs 10 \
    --batch_size 512 \
    --lr 1e-4

echo ""
echo "Step 3: CNN-MoE对抗训练"
echo "----------------------------------------"
python gan/adversarial_training.py \
    --dataset_path ${DATASET_PATH} \
    --adversarial_path ${ADV_SAMPLES_PATH} \
    --model_type cnn_moe \
    --pretrained_path ${CNN_MOE_PRETRAINED} \
    --output_path ${CNN_MOE_ADV_OUTPUT} \
    --adv_ratio ${ADV_RATIO} \
    --epochs 10 \
    --batch_size 512 \
    --lr 1e-4

echo ""
echo "Step 4: 评估ASR变化（对抗训练前后对比）"
echo "----------------------------------------"
echo "4.1 评估原始CNN模型的ASR"
python gan/attack_model.py \
    --target_model_path ${CNN_PRETRAINED} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "4.2 评估对抗训练后CNN模型的ASR"
python gan/attack_model.py \
    --target_model_path ${CNN_ADV_OUTPUT} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "4.3 评估原始CNN-MoE模型的ASR"
python gan/attack_model.py \
    --target_model_path ${CNN_MOE_PRETRAINED} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "4.4 评估对抗训练后CNN-MoE模型的ASR"
python gan/attack_model.py \
    --target_model_path ${CNN_MOE_ADV_OUTPUT} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "=========================================="
echo "✓ 对抗训练流程完成！"
echo "=========================================="
echo ""
echo "模型保存位置："
echo "  - CNN对抗训练模型: ${CNN_ADV_OUTPUT}"
echo "  - CNN-MoE对抗训练模型: ${CNN_MOE_ADV_OUTPUT}"
echo ""
echo "请查看ASR变化来验证对抗训练效果"
echo "期望：对抗训练后ASR显著下降，真实测试集Acc保持或提升"
