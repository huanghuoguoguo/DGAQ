import torch
import torch.nn as nn
import torch.nn.functional as F

class DGAGenerator(nn.Module):
    """
    改进的自回归 DGA 域名生成器
    Generator: z (Noise) -> Domain Sequence (Softmax Probabilities)
    
    改进点：
    1. 自回归生成（类似seq2seq decoder）
    2. 支持EOS终止机制
    3. 单一噪声向量初始化隐状态
    """
    def __init__(self, vocab_size, hidden_dim=256, embedding_dim=64, max_len=60, z_dim=100):
        super(DGAGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # PAD和EOS索引（假设PAD=vocab_size-2, EOS=vocab_size-1）
        self.pad_idx = vocab_size - 2
        self.eos_idx = vocab_size - 1
        self.start_idx = self.pad_idx  # START 使用 PAD 索引，避免偏向具体字符
        
        # 噪声z映射到LSTM初始隐状态和细胞状态
        self.fc_hidden = nn.Linear(z_dim, hidden_dim)
        self.fc_cell = nn.Linear(z_dim, hidden_dim)
        
        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 自回归LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # 输出层：Hidden -> Vocab Probabilities
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, batch_size, device, temperature=1.0):
        """
        自回归生成域名序列
        :param batch_size: 批量大小
        :param device: 设备
        :param temperature: Gumbel-Softmax温度参数
        :return: (batch_size, max_len, vocab_size) 的概率分布
        """
        # 生成随机噪声向量 z (每个样本一个噪声向量)
        z = torch.randn(batch_size, self.z_dim).to(device)
        
        # 初始化LSTM隐状态和细胞状态（2层）
        h0 = self.fc_hidden(z).unsqueeze(0).repeat(2, 1, 1)  # (2, batch, hidden_dim)
        c0 = self.fc_cell(z).unsqueeze(0).repeat(2, 1, 1)
        
        # 存储每个时间步的输出概率
        outputs = []
        
        # 初始输入：随机起始字符（避开PAD/EOS）
        input_token = torch.randint(0, self.vocab_size - 2, (batch_size,), dtype=torch.long).to(device)
        
        # 自回归生成
        for t in range(self.max_len):
            # 嵌入当前输入token
            emb = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embedding_dim)
            
            # LSTM前向传播
            lstm_out, (h0, c0) = self.lstm(emb, (h0, c0))
            
            # 映射到词汇表大小
            logits = self.out(lstm_out.squeeze(1))  # (batch, vocab_size)
            
            # 训练前向：不使用EOS/PAD约束，仅做温度采样
            probs = F.softmax(logits / 0.8, dim=-1)
            
            outputs.append(probs)
            
            # 下一步输入：训练用argmax，推理用采样
            if self.training:
                # 训练时：使用Gumbel-Softmax的soft输出
                input_token = torch.argmax(probs, dim=-1)
            else:
                # 推理时：直接采样
                input_token = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        
        # 拼接所有时间步的输出
        output_probs = torch.stack(outputs, dim=1)  # (batch, max_len, vocab_size)
        
        return output_probs

    def sample(self, batch_size, device, max_len=None, use_eos=True):
        """
        用于推理阶段，生成具体的字符索引
        支持EOS提前终止
        """
        if max_len is None:
            max_len = self.max_len
            
        with torch.no_grad():
            # 生成噪声
            z = torch.randn(batch_size, self.z_dim).to(device)
            h0 = self.fc_hidden(z).unsqueeze(0).repeat(2, 1, 1)
            c0 = self.fc_cell(z).unsqueeze(0).repeat(2, 1, 1)
            
            # 存储生成的索引
            generated = []
            input_token = torch.randint(0, self.vocab_size - 2, (batch_size,), dtype=torch.long).to(device)
            
            # 标记是否已经生成EOS
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
            
            for t in range(max_len):
                emb = self.embedding(input_token).unsqueeze(1)
                lstm_out, (h0, c0) = self.lstm(emb, (h0, c0))
                logits = self.out(lstm_out.squeeze(1))
                
                # 约束特殊符号，避免早期结束或填充
                logits[:, self.pad_idx] = -1e9
                min_len = 10
                if use_eos and t < min_len:
                    logits[:, self.eos_idx] = -1e9
                elif use_eos and t >= min_len:
                    logits[:, self.eos_idx] = logits[:, self.eos_idx] + 0.1 * (t - min_len)
                
                # 温度采样
                probs = F.softmax(logits / 0.8, dim=-1)
                
                # 对于已完成的序列，强制输出PAD
                if use_eos and finished.any():
                    probs[finished, :] = 0
                    probs[finished, self.pad_idx] = 1.0
                
                # 采样
                input_token = torch.multinomial(probs, 1).squeeze(-1)
                generated.append(input_token)
                
                # 检查是否生成了EOS
                if use_eos:
                    finished = finished | (input_token == self.eos_idx)
                    if finished.all():
                        break
            
            # 拼接生成的序列
            indices = torch.stack(generated, dim=1)  # (batch, seq_len)
            
            # 如果序列长度不足max_len，用PAD填充
            if indices.size(1) < max_len:
                pad_len = max_len - indices.size(1)
                padding = torch.full((batch_size, pad_len), self.pad_idx, dtype=torch.long).to(device)
                indices = torch.cat([indices, padding], dim=1)
                
        return indices
