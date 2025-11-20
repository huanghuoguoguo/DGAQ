import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CBAM1D(nn.Module):
    """一维CBAM注意力模块"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool1d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction_ratio, channels, 1)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_avg_pool(x)
        max_out = self.channel_max_pool(x)
        channel_att = self.sigmoid(self.channel_mlp(avg_out) + self.channel_mlp(max_out))
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class PyramidBlock(nn.Module):
    """DPCNN金字塔块结构"""
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        residual = self.pool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return residual + self.pool(x)


class LongContextExtractor(nn.Module):
    """长距离上下文特征提取器（并行CNN+金字塔）"""
    def __init__(self, d_model, num_filters=128):
        super().__init__()
        # 并行卷积分支 k=2 和 k=3
        self.conv_k2 = nn.Sequential(
            nn.Conv1d(d_model, num_filters, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        )
        
        self.conv_k3 = nn.Sequential(
            nn.Conv1d(d_model, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        )
        
        # CBAM模块
        self.cbam_k2 = CBAM1D(num_filters)
        self.cbam_k3 = CBAM1D(num_filters)
        
        # 金字塔块
        self.pyramid_k2 = PyramidBlock(num_filters)
        self.pyramid_k3 = PyramidBlock(num_filters)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model] -> [batch, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # k=2分支
        k2 = self.conv_k2(x)
        k2 = self.cbam_k2(k2)
        
        # k=3分支
        k3 = self.conv_k3(x)
        k3 = self.cbam_k3(k3)
        
        # 金字塔结构直到序列长度为1
        while k2.size(-1) > 1:
            k2 = self.pyramid_k2(k2)
        while k3.size(-1) > 1:
            k3 = self.pyramid_k3(k3)
        
        return k2.squeeze(-1), k3.squeeze(-1)  # [batch, num_filters]


class ShallowSpatialTemporalExtractor(nn.Module):
    """浅层时空特征提取器（BiLSTM+自注意力）"""
    def __init__(self, d_model, hidden_size=128):
        super().__init__()
        self.bilstm = nn.LSTM(d_model, hidden_size, bidirectional=True, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
        
    def forward(self, x):
        # BiLSTM提取特征
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, hidden*2]
        
        # 自注意力机制
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        
        # 时间维度平均池化
        return attn_out.mean(dim=1)  # [batch, hidden_size*2]


class DGA_DETECTION_MODEL(nn.Module):
    """基于Transformer和多特征融合的DGA域名检测模型"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_classes=41, max_len=40):
        super().__init__()
        
        # 1. 输入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, d_model)
        
        # 2. Transformer编码器（单层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. 特征提取层
        self.long_context_extractor = LongContextExtractor(d_model)
        self.shallow_extractor = ShallowSpatialTemporalExtractor(d_model)
        
        # 4. 输出层
        total_feature_dim = 128 * 4  # k2+k3+shallow1+shallow2
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        x: [batch, seq_len] 域名字符编码序列
        """
        # 输入层：嵌入 + 位置编码
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(pos_ids)
        
        # Transformer编码器捕获全局信息
        x = self.transformer(x)
        
        # 多特征提取
        f2_depth, f3_depth = self.long_context_extractor(x)  # 长距离上下文特征
        shallow_features = self.shallow_extractor(x)  # 浅层时空特征
        f2_shallow = shallow_features[:, :128]
        f3_shallow = shallow_features[:, 128:]
        
        # 特征融合
        fused_features = torch.cat([f2_depth, f3_depth, f2_shallow, f3_shallow], dim=1)
        
        # 分类输出
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        
        return logits


# 模型使用示例
if __name__ == '__main__':
    # 二分类示例
    model_binary = DGA_DETECTION_MODEL(vocab_size=38, num_classes=2)  # 37个字符+pad+unk
    
    # 多分类示例（40个DGA家族+良性域名）
    model_multiclass = DGA_DETECTION_MODEL(vocab_size=38, num_classes=41)
    
    # 测试前向传播
    batch_size, seq_len = 256, 40
    dummy_input = torch.randint(1, 37, (batch_size, seq_len))  # 随机域名编码
    
    with torch.no_grad():
        output = model_multiclass(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")  # [256, 41]
        print(f"预测类别: {torch.argmax(output, dim=1)[:5]}")