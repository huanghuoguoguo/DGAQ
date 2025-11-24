"""
GAN训练结果分析脚本
用于生成样本、评估质量并可视化结果
"""
import torch
import numpy as np
from core.adversarial.generator import DGAGenerator
from core.dataset import create_data_loaders
import argparse

# 字符映射表（需要与dataset.py保持一致）
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
PAD_IDX = len(CHARS)
UNK_IDX = len(CHARS) + 1

def indices_to_domain(indices):
    """将索引序列转换为域名字符串"""
    domain = ""
    for idx in indices:
        if idx == PAD_IDX or idx >= len(CHARS):
            break
        domain += IDX_TO_CHAR.get(idx, "?")
    return domain

def analyze_generated_samples(generator, device, num_samples=50):
    """生成并分析样本"""
    print(f"\n{'='*80}")
    print(f"🎲 生成 {num_samples} 个DGA域名样本")
    print(f"{'='*80}\n")
    
    # 生成样本
    indices = generator.sample(num_samples, device)
    indices_np = indices.cpu().numpy()
    
    generated_domains = []
    lengths = []
    
    print("生成的域名示例（前20个）:")
    print("-" * 80)
    for i in range(min(20, num_samples)):
        domain = indices_to_domain(indices_np[i])
        generated_domains.append(domain)
        lengths.append(len(domain))
        print(f"{i+1:3d}. {domain:40s} (长度: {len(domain)})")
    
    # 统计分析
    print(f"\n{'='*80}")
    print("📊 统计分析")
    print(f"{'='*80}")
    
    all_lengths = [len(indices_to_domain(indices_np[i])) for i in range(num_samples)]
    
    print(f"平均长度: {np.mean(all_lengths):.2f}")
    print(f"最小长度: {np.min(all_lengths)}")
    print(f"最大长度: {np.max(all_lengths)}")
    print(f"标准差: {np.std(all_lengths):.2f}")
    
    # 字符分布统计
    char_counts = {c: 0 for c in CHARS}
    for domain in generated_domains:
        for c in domain:
            if c in char_counts:
                char_counts[c] += 1
    
    print(f"\n字符使用频率（Top 10）:")
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:10]:
        print(f"  '{char}': {count} 次")
    
    # 质量评估
    print(f"\n{'='*80}")
    print("🔍 质量评估")
    print(f"{'='*80}")
    
    # 1. 有效域名比例（非空且不全是特殊字符）
    valid_count = sum(1 for d in generated_domains if len(d) > 3 and any(c.isalnum() for c in d))
    print(f"有效域名比例: {valid_count}/{len(generated_domains)} ({100*valid_count/len(generated_domains):.1f}%)")
    
    # 2. 包含数字的域名比例
    with_digits = sum(1 for d in generated_domains if any(c.isdigit() for c in d))
    print(f"包含数字的域名: {with_digits}/{len(generated_domains)} ({100*with_digits/len(generated_domains):.1f}%)")
    
    # 3. 包含连字符的域名比例
    with_hyphen = sum(1 for d in generated_domains if '-' in d)
    print(f"包含连字符的域名: {with_hyphen}/{len(generated_domains)} ({100*with_hyphen/len(generated_domains):.1f}%)")
    
    # 4. 熵值分析（多样性）
    unique_domains = len(set(generated_domains))
    print(f"唯一域名数量: {unique_domains}/{len(generated_domains)} ({100*unique_domains/len(generated_domains):.1f}%)")
    
    return generated_domains

def main():
    parser = argparse.ArgumentParser(description="Analyze GAN Training Results")
    parser.add_argument('--model_path', type=str, required=True, help='Path to generator model')
    parser.add_argument('--vocab_size', type=int, default=40, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=60, help='Max sequence length')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载生成器
    print(f"\n加载生成器模型: {args.model_path}")
    generator = DGAGenerator(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        max_len=args.max_len,
        z_dim=args.z_dim
    ).to(device)
    
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    print("✅ 模型加载成功")
    
    # 分析生成样本
    generated_domains = analyze_generated_samples(generator, device, args.num_samples)
    
    # 保存结果
    output_file = "gan_generated_samples.txt"
    with open(output_file, 'w') as f:
        for domain in generated_domains:
            f.write(domain + '\n')
    print(f"\n✅ 生成的域名已保存到: {output_file}")

if __name__ == "__main__":
    main()
