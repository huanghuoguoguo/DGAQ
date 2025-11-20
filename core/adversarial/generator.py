import torch
import torch.nn as nn
import torch.nn.functional as F

class DGAGenerator(nn.Module):
    """
    基于 LSTM 的 DGA 域名生成器
    Generator: z (Noise) -> Domain Sequence (Softmax Probabilities)
    """
    def __init__(self, vocab_size, hidden_dim=256, embedding_dim=32, max_len=60, z_dim=100):
        super(DGAGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # 将噪声 z 映射到 LSTM 的初始隐藏状态
        self.linear = nn.Linear(z_dim, hidden_dim)
        
        # LSTM 层
        # input_size 是 hidden_dim，因为我们在每个时间步输入的是上一步的输出（经过embedding）
        # 这里为了简化，我们直接用 hidden_state 推进，或者使用 standard seq2seq decoder approach
        # 但对于 GAN 生成文本，通常做法是：
        # 1. 初始状态由 z 产生
        # 2. 输入可以是 Start Token 或者 这里的 z 扩展
        
        # 简化版架构：
        # z -> Linear -> Reshape -> LSTM Input (repeated)
        
        self.lstm = nn.LSTM(z_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # 输出层：Hidden -> Vocab Probabilities
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, batch_size, device):
        """
        生成域名序列
        :param batch_size: 批量大小
        :param device: 设备
        :return: (batch_size, max_len, vocab_size) 的概率分布
        """
        # 生成随机噪声 z
        z = torch.randn(batch_size, self.max_len, self.z_dim).to(device)
        
        # LSTM 前向传播
        # input: (batch, seq_len, z_dim)
        lstm_out, _ = self.lstm(z)
        
        # 映射到词汇表大小
        # output: (batch, seq_len, vocab_size)
        logits = self.out(lstm_out)
        
        # 使用 Gumbel-Softmax 保证可微性，或者直接返回 Softmax
        # 在 WGAN-GP 中，通常直接输出 Softmax 概率分布传给 Discriminator
        probs = F.softmax(logits, dim=2)
        
        return probs

    def sample(self, batch_size, device):
        """
        用于推理阶段，生成具体的字符索引
        """
        with torch.no_grad():
            probs = self.forward(batch_size, device)
            # 取概率最大的字符索引
            indices = torch.argmax(probs, dim=2)
        return indices
