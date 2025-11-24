#!/bin/bash
# 完整的混合多epoch对抗训练实验流程

set -e

echo "=========================================="
echo "混合多Epoch对抗训练完整实验"
echo "=========================================="
echo ""

# 等待CNN训练完成后，继续CNN-MoE训练
echo "Step 1: CNN-MoE混合对抗训练"
echo "----------------------------------------"
python gan/adversarial_training.py \
    --dataset_path ./data/processed/500k_unified_dga_dataset.pkl \
    --adversarial_path ./data/processed/mixed_adversarial_samples.pkl \
    --model_type cnn_moe \
    --pretrained_path ./models/cnn_moe_binary_model.pth \
    --output_path ./models/cnn_moe_mixed_adversarial_trained.pth \
    --adv_ratio 0.30 \
    --epochs 10 \
    --batch_size 512 \
    --lr 1e-4

echo ""
echo "Step 2: 评估混合对抗训练的泛化能力"
echo "----------------------------------------"

# 测试混合训练模型对各epoch的ASR
epochs=(5 10 20 30 40 50)

echo "对比：单一Epoch5训练 vs 混合训练"
echo ""

for epoch in "${epochs[@]}"; do
    echo "测试生成器 Epoch $epoch:"
    
    echo "  原始CNN:"
    python gan/attack_model.py \
        --target_model_path ./models/cnn_binary_model.pth \
        --generator_path ./models/gan/generator_epoch_${epoch}.pth \
        --num_samples 1000 | grep "ASR:"
    
    echo "  单一E5对抗训练CNN:"
    python gan/attack_model.py \
        --target_model_path ./models/cnn_adversarial_trained.pth \
        --generator_path ./models/gan/generator_epoch_${epoch}.pth \
        --num_samples 1000 | grep "ASR:"
    
    echo "  混合对抗训练CNN:"
    python gan/attack_model.py \
        --target_model_path ./models/cnn_mixed_adversarial_trained.pth \
        --generator_path ./models/gan/generator_epoch_${epoch}.pth \
        --num_samples 1000 | grep "ASR:"
    
    echo ""
done

echo "=========================================="
echo "✓ 混合对抗训练实验完成！"
echo "=========================================="
