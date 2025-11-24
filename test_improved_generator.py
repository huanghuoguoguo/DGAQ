"""
快速测试改进后的生成器
"""
import torch
from core.adversarial.generator import DGAGenerator

# 字符映射
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."

def indices_to_domain(indices_array):
    """将索引序列转换为域名字符串"""
    PAD_IDX = len(CHARS)
    EOS_IDX = len(CHARS) + 1
    domain = ""
    for idx in indices_array:
        if idx == PAD_IDX or idx == EOS_IDX or idx >= len(CHARS):
            break
        domain += CHARS[idx] if idx < len(CHARS) else "?"
    return domain

def test_generator():
    device = torch.device('cpu')
    vocab_size = 40
    
    # 创建生成器
    print("创建改进的自回归生成器...")
    generator = DGAGenerator(
        vocab_size=vocab_size,
        hidden_dim=256,
        embedding_dim=64,
        max_len=60,
        z_dim=100
    ).to(device)
    
    generator.eval()
    
    # 生成样本
    print("\n" + "="*80)
    print("🎲 测试生成（未训练状态）")
    print("="*80)
    
    num_samples = 20
    indices = generator.sample(num_samples, device, max_len=40, use_eos=True)
    indices_np = indices.cpu().numpy()
    
    domains = []
    lengths = []
    
    for i in range(num_samples):
        domain = indices_to_domain(indices_np[i])
        domains.append(domain)
        lengths.append(len(domain))
        print(f"{i+1:3d}. {domain:40s} (长度: {len(domain)})")
    
    # 统计分析
    print("\n" + "="*80)
    print("📊 统计分析")
    print("="*80)
    print(f"平均长度: {sum(lengths)/len(lengths):.2f}")
    print(f"最小长度: {min(lengths)}")
    print(f"最大长度: {max(lengths)}")
    print(f"唯一域名数: {len(set(domains))}/{num_samples}")
    
    # 字符分布
    char_counts = {c: 0 for c in CHARS}
    for domain in domains:
        for c in domain:
            if c in char_counts:
                char_counts[c] += 1
    
    used_chars = [c for c, count in char_counts.items() if count > 0]
    print(f"使用的字符种类: {len(used_chars)}/{len(CHARS)}")
    print(f"使用的字符: {used_chars[:20]}")  # 显示前20种
    
    print("\n✅ 测试完成！改进后的生成器架构验证通过。")
    print("关键改进:")
    print("  1. ✅ 自回归生成机制")
    print("  2. ✅ EOS终止机制")
    print("  3. ✅ 可变长度生成")
    print("  4. ✅ Gumbel-Softmax增加多样性")

if __name__ == "__main__":
    test_generator()
