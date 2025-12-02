import torch
import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader
from core.dataset import create_data_loaders, DGADataset
from core.adversarial.gan_trainer import WGAN_GP_Trainer

def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP for DGA Generation")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--dataset_path', type=str, default='./data/processed/500k_unified_dga_dataset.pkl', help='Path to dataset')
    # 新增：判别器类型与mamba2嵌入维度
    parser.add_argument('--discriminator', type=str, default='cnn', choices=['mamba2', 'cnn'], help='Discriminator type: mamba2 or cnn')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dim for Mamba2 discriminator (must be multiple of 8)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Dataset
    print("Loading dataset...")
    try:
        # We use 'binary' task type to get simple labels, but for GAN we mainly need the sequences (X)
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            task_type='binary'
        )
    except FileNotFoundError:
        print(f"Dataset not found at {args.dataset_path}. Using synthetic data for sanity check.")
        # Synthetic fallback
        vocab_size = 40
        max_len = 60
        num_classes = 2
        num_samples = 5000
        import numpy as np
        X = np.random.randint(0, vocab_size, size=(num_samples, max_len))
        y = np.random.randint(0, num_classes, size=(num_samples,))
        full_dataset = DGADataset(X, y)
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        test_size = num_samples - train_size - val_size
        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size)
        dataset_info = {
            'vocab_size': vocab_size,
            'max_length': max_len,
            'num_classes': num_classes,
            'total_samples': num_samples
        }
    
    # TODO: Filter for DGA domains only if the goal is to mimic DGA families
    # Currently using all data (Benign + Malicious) which might confuse the GAN
    # Let's assume we want to generate "DGA-like" domains, so we should probably train on DGA samples.
    # However, if we want to generate "Adversarial" samples that look benign, we might want to train on Benign?
    # Usually, to make a robust classifier, we want to generate samples that are *hard* for the classifier.
    # If we train on DGA, we generate more DGA.
    # Let's stick to the dataset provided.
    
    config = {
        'vocab_size': dataset_info['vocab_size'],
        'max_length': dataset_info['max_length'], # Note: dataset.py uses 'max_length'
        'max_len': dataset_info['max_length'],    # Trainer uses 'max_len'
        'batch_size': args.batch_size,
        'lr': args.lr,
        'z_dim': args.z_dim,
        'hidden_dim': args.hidden_dim,
        'lambda_gp': 10,
        'n_critic': 5,
        # 新增：判别器与mamba2参数
        'discriminator': args.discriminator,
        'embedding_dim': args.embedding_dim
    }
    
    print(f"Dataset Info: Vocab Size={config['vocab_size']}, Max Len={config['max_len']}")

    trainer = WGAN_GP_Trainer(config, device)
    trainer.train(train_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
