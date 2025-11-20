import torch
import argparse
import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.model.mamba2_moe_model import LightweightMamba2MoE
from core.adversarial.generator import DGAGenerator
from core.dataset import load_dataset

def load_mamba_model(model_path, device, dataset_info):
    """Load the target Mamba2-MoE model"""
    model = LightweightMamba2MoE(
        vocab_size=dataset_info['vocab_size'],
        embedding_dim=256,
        max_length=dataset_info['max_length'],
        num_classes=dataset_info['num_classes'],
        num_layers=2,       # Default config from train_mamba2_moe.py
        num_experts=3,
        d_state=128,
        headdim=64
    ).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading Mamba2-MoE model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights for testing.")
    
    model.eval()
    return model

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

def attack(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Get Dataset Info (for vocab size etc)
    # We need to know the vocab mapping to decode or to ensure compatibility
    # For simplicity, we assume the dataset pickle has the info
    dataset = load_dataset(args.dataset_path)
    if 'info' in dataset:
        dataset_info = dataset['info']
    elif 'metadata' in dataset:
        dataset_info = dataset['metadata']
    else:
        # Fallback defaults
        dataset_info = {
            'vocab_size': 40, # Approximate
            'max_length': 60,
            'num_classes': 2
        }
    
    print(f"Dataset Info: Vocab={dataset_info.get('vocab_size')}, MaxLen={dataset_info.get('max_length')}")

    # 2. Load Models
    target_model = load_mamba_model(args.target_model_path, device, dataset_info)
    
    gen_config = {
        'vocab_size': dataset_info.get('vocab_size', 40),
        'hidden_dim': args.hidden_dim,
        'max_len': dataset_info.get('max_length', 60),
        'z_dim': args.z_dim
    }
    generator = load_generator(args.generator_path, device, gen_config)
    
    # 3. Generate Adversarial Samples
    print(f"Generating {args.num_samples} adversarial samples...")
    with torch.no_grad():
        # Generator outputs probabilities or indices?
        # The Generator.sample() method returns indices
        adv_indices = generator.sample(args.num_samples, device) # (N, L)
        
        # 4. Attack Target Model
        # Target model expects indices (B, L)
        logits = target_model(adv_indices)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        # 5. Calculate Success Rate
        # We generated these samples, so they ARE DGA (Malicious).
        # If the classifier predicts 0 (Benign), the attack is successful.
        # If the classifier predicts 1 (Malicious), the attack failed (detected).
        
        # Assuming 0 = Benign, 1 = Malicious
        successful_evasions = (predictions == 0).sum().item()
        asr = successful_evasions / args.num_samples
        
        print("\n" + "="*30)
        print(f"Attack Results:")
        print(f"Total Samples: {args.num_samples}")
        print(f"Evasions (Classified as Benign): {successful_evasions}")
        print(f"Detections (Classified as Malicious): {args.num_samples - successful_evasions}")
        print(f"Attack Success Rate (ASR): {asr*100:.2f}%")
        print("="*30 + "\n")
        
        # Show some examples
        print("Example Evasions (Indices):")
        if successful_evasions > 0:
            evasion_indices = torch.where(predictions == 0)[0][:5]
            for idx in evasion_indices:
                print(f"  {adv_indices[idx].cpu().numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack Mamba2-MoE with GAN")
    parser.add_argument('--target_model_path', type=str, default='./models/mamba2_moe_best_binary.pth')
    parser.add_argument('--generator_path', type=str, default='./models/gan/generator_epoch_50.pth')
    parser.add_argument('--dataset_path', type=str, default='./data/processed/small_dga_dataset.pkl')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    args = parser.parse_args()
    attack(args)
