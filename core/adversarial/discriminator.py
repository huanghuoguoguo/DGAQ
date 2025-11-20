import torch
import torch.nn as nn

class DGADiscriminator(nn.Module):
    """
    基于 1D CNN 的 DGA 判别器 (Critic)
    Discriminator: Domain Sequence (One-Hot) -> Real/Fake Score
    """
    def __init__(self, vocab_size, hidden_dim=256, max_len=60):
        super(DGADiscriminator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # 输入是 One-Hot 编码的序列: (batch, vocab_size, seq_len)
        # 或者 (batch, seq_len, vocab_size) -> 需要 transpose
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(vocab_size, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 全连接层输出分数 (WGAN 不需要 Sigmoid)
        self.fc = nn.Linear(hidden_dim * max_len, 1)

    def forward(self, x):
        """
        :param x: (batch, max_len, vocab_size) - One-Hot or Softmax Probabilities
        :return: (batch, 1) Score
        """
        # 调整维度以适配 Conv1d: (batch, vocab_size, max_len)
        x = x.transpose(1, 2)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 输出分数
        score = self.fc(out)
        return score
