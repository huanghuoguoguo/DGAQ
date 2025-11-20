#!/usr/bin/env python3
"""测试MoE优化前后的性能对比"""
import torch
import time
import sys
sys.path.insert(0, '/jsj_ywj/yhh/DGAQ')

from core.model.mamba2_moe_model import LightweightMamba2MoE

device = torch.device('cuda')
print(f"使用设备: {device}")

# 测试参数
B, L, V, C = 32, 60, 40, 2  # batch_size=32模拟实际训练
x = torch.randint(0, V, (B, L)).to(device)
y = torch.randint(0, C, (B,)).to(device)

print("\n创建Mamba2-MoE模型...")
model = LightweightMamba2MoE(
    vocab_size=V,
    embedding_dim=256,
    num_classes=C,
    num_layers=2,
    num_experts=3,
    d_state=128,
    headdim=64,
    balance_weight=1e-2
).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 预热
print("\n预热GPU...")
for _ in range(5):
    logits, info = model(x, return_gate_info=True)
    loss = model.compute_loss(logits, y, info)

# 性能测试 - 前向传播
print("\n测试前向传播速度...")
model.eval()
torch.cuda.synchronize()
start = time.time()
num_iters = 100
with torch.no_grad():
    for _ in range(num_iters):
        logits, info = model(x, return_gate_info=True)
torch.cuda.synchronize()
forward_time = time.time() - start
print(f"✅ {num_iters}次前向传播: {forward_time:.3f}s")
print(f"✅ 平均耗时: {forward_time/num_iters*1000:.2f}ms/batch")
print(f"✅ 吞吐量: {num_iters*B/forward_time:.1f} samples/s")

# 性能测试 - 前向+反向传播
print("\n测试前向+反向传播速度...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.cuda.synchronize()
start = time.time()
num_iters = 50
for _ in range(num_iters):
    optimizer.zero_grad()
    logits, info = model(x, return_gate_info=True)
    loss = model.compute_loss(logits, y, info)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
train_time = time.time() - start
print(f"✅ {num_iters}次训练迭代: {train_time:.3f}s")
print(f"✅ 平均耗时: {train_time/num_iters*1000:.2f}ms/iter")
print(f"✅ 训练吞吐量: {num_iters*B/train_time:.1f} samples/s")

# 检查GPU利用率
print("\n专家负载均衡:")
logits, info = model(x, return_gate_info=True)
usage = info['expert_usage'].mean(dim=0)
for i, u in enumerate(usage):
    print(f"  专家{i}: {u.item()*100:.1f}%")

print("\n✅ 测试完成！")
print("\n预期性能（batch_size=32）:")
print("  - 前向传播: ~20-40ms/batch")
print("  - 训练迭代: ~40-80ms/iter") 
print("  - 训练吞吐量: ~400-800 samples/s")
