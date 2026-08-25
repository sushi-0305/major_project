import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

from models.prototypical_networks import PrototypicalNetworks, ConvNet, PrototypicalNetworksTrainer
from models.siamese_networks import SiameseNetwork, SiameseNetworkTrainer
from utils.dataset import MedicalImageDataset, FewShotDataLoader, DataScarcitySimulator
from utils.data_augmentation import DataAugmentation
from utils.evaluation import PerformanceDegradationAnalyzer, FewShotEvaluator


def create_synthetic_dataset(num_classes=5, samples_per_class=100, image_size=84, num_channels=3):
    """Create synthetic medical image dataset for demonstration"""
    images = np.random.randn(num_classes * samples_per_class, num_channels, image_size, image_size).astype(np.float32)
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    
    # Normalize to [0, 1]
    images = (images - images.min()) / (images.max() - images.min())
    
    return torch.from_numpy(images), torch.from_numpy(labels)


def train_prototypical_networks(args):
    """Train Prototypical Networks"""
    print("Training Prototypical Networks...")
    
    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    images, labels = create_synthetic_dataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size
    )
    
    # Create model
    backbone = ConvNet(in_channels=3, hidden_dim=64)
    model = PrototypicalNetworks(backbone, hidden_dim=64)
    
    # Create trainer
    trainer = PrototypicalNetworksTrainer(model, device, lr=args.lr)
    
    # Create episode loader
    dataset = MedicalImageDataset(images.numpy(), labels.numpy())
    episode_loader = FewShotDataLoader(
        dataset,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        num_episodes=args.num_episodes
    )
    
    # Training loop
    best_accuracy = 0
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Create episode loader for this epoch
        train_loader = FewShotDataLoader(
            dataset,
            n_way=args.n_way,
            n_shot=args.n_shot,
            n_query=args.n_query,
            num_episodes=args.num_episodes
        )
        
        # Convert to regular dataloader
        dummy_loader = [(images[i*100:(i+1)*100].to(device), labels[i*100:(i+1)*100].to(device)) 
                       for i in range(len(images) // 100)]
        
        train_loss, train_acc = trainer.train_epoch(dummy_loader, args.n_way, args.n_shot, args.n_query)
        
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc
        })
        
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), f"{args.save_dir}/proto_networks_best.pth")
    
    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    
    return results


def train_siamese_networks(args):
    """Train Siamese Networks"""
    print("Training Siamese Networks...")
    
    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    images, labels = create_synthetic_dataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size
    )
    
    # Create model
    model = SiameseNetwork(in_channels=3, hidden_dim=64)
    
    # Create trainer
    trainer = SiameseNetworkTrainer(model, device, lr=args.lr, margin=args.margin)
    
    # Create data loader
    dataset = MedicalImageDataset(images.numpy(), labels.numpy())
    
    dummy_loader = [(images[i*100:(i+1)*100].to(device), labels[i*100:(i+1)*100].to(device)) 
                   for i in range(len(images) // 100)]
    
    # Training loop
    best_accuracy = 0
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = trainer.train_epoch(dummy_loader, args.n_way, args.n_shot, args.n_query)
        
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc
        })
        
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), f"{args.save_dir}/siamese_networks_best.pth")
    
    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    
    return results


def analyze_data_scarcity(args):
    """Analyze model performance degradation with data scarcity"""
    print("Analyzing data scarcity impact...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    images, labels = create_synthetic_dataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size
    )
    
    dataset = MedicalImageDataset(images.numpy(), labels.numpy())
    
    # Create model
    backbone = ConvNet(in_channels=3, hidden_dim=64)
    model = PrototypicalNetworks(backbone, hidden_dim=64).to(device)
    
    analyzer = PerformanceDegradationAnalyzer()
    
    # Test with different data percentages
    for data_pct in [100, 75, 50, 25, 10, 5]:
        print(f"\nTesting with {data_pct}% of data...")
        
        if data_pct < 100:
            reduced_dataset = DataScarcitySimulator.reduce_dataset(dataset, data_pct / 100)
        else:
            reduced_dataset = dataset
        
        # Create episode loader
        episode_loader = FewShotDataLoader(
            reduced_dataset,
            n_way=args.n_way,
            n_shot=args.n_shot,
            n_query=args.n_query,
            num_episodes=100
        )
        
        # Evaluate
        results = FewShotEvaluator.evaluate_episodes(model, episode_loader, device, num_episodes=20)
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    
    return analyzer.get_summary()


def main():
    parser = argparse.ArgumentParser(description='Train Few-Shot Medical Image Classification Models')
    
    # Dataset arguments
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--samples_per_class', type=int, default=100, help='Samples per class')
    parser.add_argument('--image_size', type=int, default=84, help='Image size')
    
    # Few-shot arguments
    parser.add_argument('--n_way', type=int, default=5, help='N-way classification')
    parser.add_argument('--n_shot', type=int, default=5, help='N-shot learning')
    parser.add_argument('--n_query', type=int, default=15, help='N query samples')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive loss margin')
    parser.add_argument('--num_episodes', type=int, default=100, help='Episodes per epoch')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='prototypical', choices=['prototypical', 'siamese', 'both'],
                       help='Model to train')
    parser.add_argument('--analyze_scarcity', action='store_true', help='Analyze data scarcity impact')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {}
    
    # Train models
    if args.model in ['prototypical', 'both']:
        results['prototypical'] = train_prototypical_networks(args)
    
    if args.model in ['siamese', 'both']:
        results['siamese'] = train_siamese_networks(args)
    
    # Analyze data scarcity
    if args.analyze_scarcity:
        results['data_scarcity'] = analyze_data_scarcity(args)
    
    # Save results
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete! Results saved to", args.save_dir)


if __name__ == '__main__':
    main()
