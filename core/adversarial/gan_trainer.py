import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from .generator import DGAGenerator
from .discriminator import DGADiscriminator

class WGAN_GP_Trainer:
    """
    WGAN-GP 训练器
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # 初始化模型
        self.generator = DGAGenerator(
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            max_len=config['max_len'],
            z_dim=config['z_dim']
        ).to(device)
        
        self.discriminator = DGADiscriminator(
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            max_len=config['max_len']
        ).to(device)
        
        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config['lr'], betas=(0.5, 0.9))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.9))
        
        # 配置参数
        self.lambda_gp = config.get('lambda_gp', 10)
        self.n_critic = config.get('n_critic', 5)
        self.batch_size = config['batch_size']
        self.vocab_size = config['vocab_size']
        self.max_len = config['max_len']

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        计算梯度惩罚 (Gradient Penalty)
        """
        # 随机权重 alpha: (batch_size, 1, 1)
        alpha = torch.rand((real_samples.size(0), 1, 1)).to(self.device)
        
        # 在真实样本和生成样本之间进行插值
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        # 判别器对插值样本的输出
        d_interpolates = self.discriminator(interpolates)
        
        # 计算梯度
        fake = torch.ones((real_samples.size(0), 1)).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # 计算梯度的范数
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def train(self, dataloader, epochs=100, save_path='./models/gan'):
        """
        训练循环
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print(f"🚀 开始 WGAN-GP 训练... (Epochs: {epochs})")
        
        for epoch in range(epochs):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
            
            for i, (real_seqs, _) in pbar:
                batch_size = real_seqs.size(0)
                real_seqs = real_seqs.to(self.device)
                
                # 将索引序列转换为 One-Hot 编码 (batch, max_len, vocab_size)
                real_one_hot = F.one_hot(real_seqs, num_classes=self.vocab_size).float()
                
                # ================================================================== #
                #                      1. 训练 Discriminator (Critic)                #
                # ================================================================== #
                
                self.d_optimizer.zero_grad()
                
                # 生成假样本 (Softmax Probabilities)
                fake_probs = self.generator(batch_size, self.device)
                
                # 判别器打分
                real_validity = self.discriminator(real_one_hot)
                fake_validity = self.discriminator(fake_probs.detach()) # Detach to avoid training G
                
                # 梯度惩罚
                gradient_penalty = self.compute_gradient_penalty(real_one_hot, fake_probs.detach())
                
                # Adversarial Loss (Wasserstein Distance)
                # Minimize - (E[D(x)] - E[D(G(z))]) + lambda * GP
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
                
                d_loss.backward()
                self.d_optimizer.step()
                
                # ================================================================== #
                #                        2. 训练 Generator                           #
                # ================================================================== #
                
                # 每 n_critic 次更新一次 Generator
                if i % self.n_critic == 0:
                    self.g_optimizer.zero_grad()
                    
                    # 重新生成假样本 (保留梯度)
                    fake_probs = self.generator(batch_size, self.device)
                    
                    # 判别器打分
                    fake_validity = self.discriminator(fake_probs)
                    
                    # Generator Loss
                    # Minimize - E[D(G(z))]
                    g_loss = -torch.mean(fake_validity)
                    
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'D Loss': d_loss.item(), 
                        'G Loss': g_loss.item()
                    })
            
            # 每个 Epoch 保存一次模型
            if (epoch + 1) % 5 == 0:
                torch.save(self.generator.state_dict(), os.path.join(save_path, f'generator_epoch_{epoch+1}.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(save_path, f'discriminator_epoch_{epoch+1}.pth'))
                
                # 生成一些样本看看效果
                self.generate_samples(5)

    def generate_samples(self, num_samples=5):
        """
        生成并打印样本
        """
        indices = self.generator.sample(num_samples, self.device)
        # 这里需要一个 index_to_char 的映射，暂时打印索引或需要传入 vocab
        print(f"\n[Sample Indices]: {indices[0].cpu().numpy()}")
        # TODO: Decode to string if vocab is available
