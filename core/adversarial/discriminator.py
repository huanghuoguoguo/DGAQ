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

class Mamba2Discriminator(nn.Module):
    """
    基于 Mamba2 的 DGA 判别器 (Critic)
    接受索引序列或概率/one-hot序列，输出 WGAN-GP 所需的标量分数。
    - 若输入为 (batch, seq_len) 的 LongTensor（索引），使用 nn.Embedding 查表。
    - 若输入为 (batch, seq_len, vocab_size) 的 FloatTensor（one-hot/prob），进行“软嵌入”：x @ embedding.weight。
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 max_len: int = 60,
                 num_layers: int = 2,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 dropout_rate: float = 0.3):
        super().__init__()
        # 延迟导入，避免在未安装 mamba-ssm 时直接报错
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "mamba-ssm 未安装或版本不满足要求。请安装 mamba-ssm 并确保使用 CUDA 环境。"
            )
        if embedding_dim % 8 != 0:
            raise ValueError(f"embedding_dim 必须是 8 的倍数，当前值: {embedding_dim}")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        # 嵌入与位置编码
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim) * 0.02)
        # Mamba2 层与层归一化
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        # Critic 头输出标量分数
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        # 简单初始化
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def _embed_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入统一映射到嵌入空间:
        - 索引: [B, L] -> embedding(x) + pos
        - 概率/one-hot: [B, L, V] -> x @ embedding.weight + pos
        返回: [B, L, D]
        """
        if x.dtype == torch.long and x.dim() == 2:
            x_emb = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        elif x.dim() == 3 and x.size(2) == self.vocab_size:
            # 软嵌入，保持可微性
            x_emb = torch.matmul(x, self.embedding.weight) + self.pos_encoding[:, :x.size(1), :]
        else:
            raise ValueError("输入维度不合法，需为 [B, L] (索引) 或 [B, L, V] (one-hot/prob)")
        return self.dropout(x_emb.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统一到嵌入空间
        x = self._embed_inputs(x)
        # Mamba2 堆叠
        for m2, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(m2(x.contiguous()))
            x = residual + self.dropout(x)
        # 位置掩码（忽略 padding=0 的位置，仅在索引输入时有效）
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(-1).float()
        x = (x * padding_mask).sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1)
        # Critic 头输出分数（无 Sigmoid）
        x = self.dropout(torch.relu(self.fc1(x)))
        score = self.fc2(x)
        return score
