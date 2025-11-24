#!/bin/bash
# ASR评估：对抗训练前后对比

echo "=========================================="
echo "Step 4: 评估ASR变化（对抗训练前后对比）"
echo "=========================================="
echo ""

GENERATOR_PATH="./models/gan/generator_epoch_5.pth"
DATASET_PATH="./data/processed/500k_unified_dga_dataset.pkl"
CNN_PRETRAINED="./models/cnn_binary_model.pth"
CNN_ADV_OUTPUT="./models/cnn_adversarial_trained.pth"
CNN_MOE_PRETRAINED="./models/cnn_moe_binary_model.pth"
CNN_MOE_ADV_OUTPUT="./models/cnn_moe_adversarial_trained.pth"

echo "4.1 评估原始CNN模型的ASR"
echo "----------------------------------------"
python gan/attack_model.py \
    --target_model_path ${CNN_PRETRAINED} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "4.2 评估对抗训练后CNN模型的ASR"
echo "----------------------------------------"
python gan/attack_model.py \
    --target_model_path ${CNN_ADV_OUTPUT} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "4.3 评估原始CNN-MoE模型的ASR"
echo "----------------------------------------"
python gan/attack_model.py \
    --target_model_path ${CNN_MOE_PRETRAINED} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "4.4 评估对抗训练后CNN-MoE模型的ASR"
echo "----------------------------------------"
python gan/attack_model.py \
    --target_model_path ${CNN_MOE_ADV_OUTPUT} \
    --generator_path ${GENERATOR_PATH} \
    --dataset_path ${DATASET_PATH} \
    --num_samples 2000

echo ""
echo "=========================================="
echo "✓ ASR评估完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  - 原始模型ASR应该较高（~70-75%）"
echo "  - 对抗训练后ASR应显著下降（预期<20%）"
