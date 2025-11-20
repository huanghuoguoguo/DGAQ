import torch
import argparse
import os
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.model.mamba2_moe_model import LightweightMamba2MoE
from core.adversarial.generator import DGAGenerator
from core.logger import get_training_logger
from core.trainer import TrainerBuilder

def load_generator(model_path, device, config):
    """Load the GAN Generator"""
    generator = DGAGenerator(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        max_len=config['max_len'],
        z_dim=config['z_dim']
    ).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading Generator from {model_path}")
        generator.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Generator file {model_path} not found. Using random weights.")
    
    generator.eval()
    return generator

def main(args):
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ Error: CUDA required")
        return
    
    device = torch.device('cuda')
    logger = get_training_logger("mamba2_moe_robust", "binary")
    logger.info("Starting Adversarial Training (Robustness Fine-tuning)")
    
    # 1. Load Data
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        task_type='binary'
    )
    print_dataset_info(dataset_info)
    
    # 2. Load Target Model (Mamba2-MoE)
    model = LightweightMamba2MoE(
        vocab_size=dataset_info['vocab_size'],
        embedding_dim=256,
        max_length=dataset_info['max_length'],
        num_classes=dataset_info['num_classes'],
        num_layers=args.num_layers,
        d_state=args.d_state,
        headdim=args.headdim,
        num_experts=args.num_experts
    ).to(device)
    
    if os.path.exists(args.target_model_path):
        logger.info(f"Loading pre-trained Mamba2-MoE from {args.target_model_path}")
        model.load_state_dict(torch.load(args.target_model_path, map_location=device))
    else:
        logger.info("Pre-trained model not found, starting from scratch (not recommended for fine-tuning)")

    # 3. Load Generator
    gen_config = {
        'vocab_size': dataset_info['vocab_size'],
        'hidden_dim': args.gen_hidden_dim,
        'max_len': dataset_info['max_length'],
        'z_dim': args.z_dim
    }
    generator = load_generator(args.generator_path, device, gen_config)
    
    # 4. Setup Trainer
    # We use a smaller learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Adversarial Training Loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # --- Generate Adversarial Samples ---
            # Generate a batch of fake domains equal to a portion of the real batch (e.g., 50%)
            num_adv = int(batch_size * args.adv_ratio)
            if num_adv > 0:
                with torch.no_grad():
                    adv_data = generator.sample(num_adv, device) # (num_adv, max_len)
                
                # Label them as Malicious (1)
                adv_target = torch.ones(num_adv, dtype=torch.long).to(device)
                
                # Concatenate real and fake data
                # Note: This increases the effective batch size
                combined_data = torch.cat([data, adv_data], dim=0)
                combined_target = torch.cat([target, adv_target], dim=0)
            else:
                combined_data = data
                combined_target = target
            
            # --- Forward & Backward ---
            optimizer.zero_grad()
            
            logits, gate_info = model(combined_data, return_gate_info=True)
            loss = model.compute_loss(logits, combined_target, gate_info)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(combined_target).sum().item()
            total += combined_target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                val_loss += F.cross_entropy(logits, target).item()
                pred = logits.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        logger.info(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {100.*val_correct/val_total:.2f}%")
        
        # Save Checkpoint
        save_path = f"./models/mamba2_moe_robust_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Training for Mamba2-MoE")
    parser.add_argument('--dataset_path', type=str, default='./data/processed/small_dga_dataset.pkl')
    parser.add_argument('--target_model_path', type=str, default='./models/mamba2_moe_best_binary.pth')
    parser.add_argument('--generator_path', type=str, default='./models/gan/generator_epoch_50.pth')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adv_ratio', type=float, default=0.3, help="Ratio of adversarial samples in each batch")
    
    # Model Config (Must match trained model)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--d_state', type=int, default=128)
    parser.add_argument('--headdim', type=int, default=64)
    
    # Generator Config
    parser.add_argument('--gen_hidden_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=100)
    
    args = parser.parse_args()
    main(args)
