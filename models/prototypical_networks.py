import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm


class ConvNet(nn.Module):
    """Convolutional backbone for feature extraction"""
    def __init__(self, in_channels=3, hidden_dim=64):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        return x


class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for Few-Shot Learning"""
    def __init__(self, backbone, hidden_dim=64):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        
    def forward(self, support_images, support_labels, query_images):
        """
        Args:
            support_images: (N_way * N_shot, C, H, W)
            support_labels: (N_way * N_shot,)
            query_images: (N_way * N_query, C, H, W)
        Returns:
            logits: (N_way * N_query, N_way)
        """
        # Extract features
        support_features = self.backbone(support_images)  # (N_way * N_shot, hidden_dim, h, w)
        query_features = self.backbone(query_images)      # (N_way * N_query, hidden_dim, h, w)
        
        # Flatten spatial dimensions
        support_features = support_features.view(support_features.size(0), -1)  # (N_way * N_shot, hidden_dim * h * w)
        query_features = query_features.view(query_features.size(0), -1)        # (N_way * N_query, hidden_dim * h * w)
        
        # Compute prototypes (mean of support features per class)
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # (N_way, hidden_dim * h * w)
        
        # Compute distances from query samples to prototypes
        distances = torch.cdist(query_features, prototypes)  # (N_way * N_query, N_way)
        
        # Convert distances to logits (negative distances = similarity)
        logits = -distances
        return logits
    
    def predict(self, support_images, support_labels, query_images):
        """Make predictions for query images"""
        logits = self.forward(support_images, support_labels, query_images)
        predictions = logits.argmax(dim=1)
        return predictions


class PrototypicalNetworksTrainer:
    """Trainer for Prototypical Networks"""
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader, n_way, n_shot, n_query):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Sample support and query sets
            support_images, support_labels, query_images, query_labels = self._sample_episode(
                images, labels, n_way, n_shot, n_query
            )
            
            # Forward pass
            logits = self.model(support_images, support_labels, query_images)
            loss = self.criterion(logits, query_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == query_labels).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        return avg_loss, avg_accuracy
    
    def evaluate(self, dataloader, n_way, n_shot, n_query):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Sample support and query sets
                support_images, support_labels, query_images, query_labels = self._sample_episode(
                    images, labels, n_way, n_shot, n_query
                )
                
                # Forward pass
                logits = self.model(support_images, support_labels, query_images)
                loss = self.criterion(logits, query_labels)
                
                # Compute accuracy
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == query_labels).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        return avg_loss, avg_accuracy
    
    def _sample_episode(self, images, labels, n_way, n_shot, n_query):
        """Sample a few-shot episode"""
        unique_labels = torch.unique(labels)
        
        # Randomly select n_way classes
        selected_labels = unique_labels[torch.randperm(len(unique_labels))[:n_way]]
        
        support_images = []
        support_labels_list = []
        query_images = []
        query_labels_list = []
        
        for class_idx, class_label in enumerate(selected_labels):
            mask = labels == class_label
            class_images = images[mask]
            
            # Randomly select n_shot + n_query samples
            indices = torch.randperm(len(class_images))[:n_shot + n_query]
            support_indices = indices[:n_shot]
            query_indices = indices[n_shot:]
            
            support_images.append(class_images[support_indices])
            support_labels_list.extend([class_idx] * n_shot)
            
            query_images.append(class_images[query_indices])
            query_labels_list.extend([class_idx] * n_query)
        
        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.tensor(support_labels_list, dtype=torch.long, device=self.device)
        query_images = torch.cat(query_images, dim=0)
        query_labels = torch.tensor(query_labels_list, dtype=torch.long, device=self.device)
        
        return support_images, support_labels, query_images, query_labels
