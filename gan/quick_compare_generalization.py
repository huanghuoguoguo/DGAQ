#!/usr/bin/env python
"""快速对比单一Epoch5训练 vs 混合训练的泛化能力"""

import subprocess
import sys

def get_asr(model_path, gen_path):
    """获取ASR"""
    cmd = [
        'python', 'gan/attack_model.py',
        '--target_model_path', model_path,
        '--generator_path', gen_path,
        '--num_samples', '1000'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        for line in result.stdout.split('\n'):
            if 'Attack Success Rate (ASR)' in line:
                return float(line.split(':')[1].strip().rstrip('%'))
        return None
    except:
        return None

print("="*80)
print("泛化能力对比：单一Epoch5训练 vs 混合多Epoch训练")
print("="*80)
print()

epochs = [5, 10, 20, 30, 40, 50]

# 表头
print(f"{'生成器':<12} {'原始CNN':>12} {'E5训练':>12} {'混合训练':>12} {'改进':>12}")
print("-"*80)

total_improvement = 0
count = 0

for epoch in epochs:
    gen_path = f"./models/gan/generator_epoch_{epoch}.pth"
    
    # 获取三个模型的ASR
    original = get_asr("./models/cnn_binary_model.pth", gen_path)
    single = get_asr("./models/cnn_adversarial_trained.pth", gen_path)
    mixed = get_asr("./models/cnn_mixed_adversarial_trained.pth", gen_path)
    
    if original and single and mixed:
        improvement = single - mixed
        total_improvement += improvement
        count += 1
        
        print(f"Epoch {epoch:<6} {original:>11.1f}% {single:>11.1f}% {mixed:>11.1f}% {improvement:>11.1f}%")

if count > 0:
    avg_improvement = total_improvement / count
    print("-"*80)
    print(f"{'平均':12} {' '*12} {' '*12} {' '*12} {avg_improvement:>11.1f}%")
    print()
    print(f"结论: 混合训练比单一训练平均降低ASR {avg_improvement:.1f}%")
    if avg_improvement > 10:
        print("✅ 混合训练显著提升了泛化能力！")
    elif avg_improvement > 5:
        print("⚠️  混合训练有一定提升")
    else:
        print("❌ 混合训练提升不明显")
