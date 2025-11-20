import torch
import argparse
from core.dataset import create_data_loaders
from core.adversarial.gan_trainer import WGAN_GP_Trainer

def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP for DGA Generation")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--dataset_path', type=str, default='./data/processed/small_dga_dataset.pkl', help='Path to dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Dataset
    print("Loading dataset...")
    # We use 'binary' task type to get simple labels, but for GAN we mainly need the sequences (X)
    # If we want to train on MALICIOUS domains only (to generate DGA), we should filter.
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        task_type='binary'
    )
    
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
        'n_critic': 5
    }
    
    print(f"Dataset Info: Vocab Size={config['vocab_size']}, Max Len={config['max_len']}")

    trainer = WGAN_GP_Trainer(config, device)
    trainer.train(train_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
