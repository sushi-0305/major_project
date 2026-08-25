import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm


class SiameseNetwork(nn.Module):
    """Siamese Network for Few-Shot Learning"""
    def __init__(self, in_channels=3, hidden_dim=64):
        super(SiameseNetwork, self).__init__()
        
        # Shared backbone
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        
        # Compute feature map size after convolutions
        # Assuming input (C, 84, 84) -> (hidden_dim, 5, 5) after 4 pools
        self.feature_size = hidden_dim * 5 * 5
        
        # Similarity head
        self.fc1 = nn.Linear(self.feature_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward_one(self, x):
        """Extract features from one image"""
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x1, x2):
        """Compute similarity between two images"""
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        combined = torch.cat([feat1, feat2], dim=1)
        similarity = self.fc1(combined)
        similarity = self.relu(similarity)
        similarity = self.fc2(similarity)
        similarity = self.relu(similarity)
        similarity = torch.sigmoid(self.fc3(similarity))
        
        return similarity
    
    def predict(self, support_images, support_labels, query_images):
        """
        Predict class for query images based on support set
        Args:
            support_images: (N_way * N_shot, C, H, W)
            support_labels: (N_way * N_shot,)
            query_images: (N_way * N_query, C, H, W)
        Returns:
            predictions: (N_way * N_query,)
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for query_img in query_images:
                similarities = []
                for support_img in support_images:
                    sim = self.forward(query_img.unsqueeze(0), support_img.unsqueeze(0))
                    similarities.append(sim.item())
                
                # Find class based on maximum similarity
                max_idx = np.argmax(similarities)
                predicted_class = support_labels[max_idx]
                predictions.append(predicted_class)
        
        return torch.tensor(predictions, device=query_images.device)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese Networks"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output, label):
        """
        Args:
            output: similarity score between 0 and 1
            label: 1 if same class, 0 if different class
        """
        loss = (1 - label) * 0.5 * torch.pow(output, 2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - output, min=0), 2)
        return loss.mean()


class SiameseNetworkTrainer:
    """Trainer for Siamese Networks"""
    def __init__(self, model, device, lr=0.001, margin=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = ContrastiveLoss(margin=margin)
        
    def train_epoch(self, dataloader, n_way, n_shot, n_query):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Generate pairs
            pairs, pair_labels = self._generate_pairs(images, labels)
            
            # Forward pass
            similarity_scores = self.model(pairs[:, 0], pairs[:, 1])
            loss = self.criterion(similarity_scores, pair_labels.float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            predictions = (similarity_scores > 0.5).long().squeeze()
            accuracy = (predictions == pair_labels).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        return avg_loss, avg_accuracy
    
    def evaluate(self, dataloader, n_way, n_shot, n_query):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Generate pairs
                pairs, pair_labels = self._generate_pairs(images, labels)
                
                # Forward pass
                similarity_scores = self.model(pairs[:, 0], pairs[:, 1])
                loss = self.criterion(similarity_scores, pair_labels.float())
                
                # Compute accuracy
                predictions = (similarity_scores > 0.5).long().squeeze()
                accuracy = (predictions == pair_labels).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        return avg_loss, avg_accuracy
    
    def _generate_pairs(self, images, labels, num_pairs=100):
        """Generate positive and negative pairs"""
        pairs = []
        pair_labels = []
        
        unique_labels = torch.unique(labels)
        
        for _ in range(num_pairs):
            # Positive pair (same class)
            if np.random.random() > 0.5:
                class_label = unique_labels[np.random.randint(0, len(unique_labels))]
                mask = labels == class_label
                indices = torch.where(mask)[0]
                
                if len(indices) >= 2:
                    idx1, idx2 = torch.randperm(len(indices))[:2]
                    pairs.append(torch.stack([images[indices[idx1]], images[indices[idx2]]]))
                    pair_labels.append(1)
                else:
                    # Negative pair (different classes)
                    labels_list = unique_labels.tolist()
                    class1, class2 = np.random.choice(labels_list, 2, replace=False)
                    mask1 = labels == class1
                    mask2 = labels == class2
                    idx1 = torch.where(mask1)[0][0]
                    idx2 = torch.where(mask2)[0][0]
                    pairs.append(torch.stack([images[idx1], images[idx2]]))
                    pair_labels.append(0)
            else:
                # Negative pair (different classes)
                labels_list = unique_labels.tolist()
                if len(labels_list) >= 2:
                    class1, class2 = np.random.choice(labels_list, 2, replace=False)
                    mask1 = labels == class1
                    mask2 = labels == class2
                    indices1 = torch.where(mask1)[0]
                    indices2 = torch.where(mask2)[0]
                    
                    if len(indices1) > 0 and len(indices2) > 0:
                        idx1 = indices1[np.random.randint(0, len(indices1))]
                        idx2 = indices2[np.random.randint(0, len(indices2))]
                        pairs.append(torch.stack([images[idx1], images[idx2]]))
                        pair_labels.append(0)
        
        if len(pairs) > 0:
            pairs = torch.stack(pairs)
            pair_labels = torch.tensor(pair_labels, dtype=torch.long, device=self.device)
        else:
            pairs = torch.zeros((1, 2, *images.shape[1:]), device=self.device)
            pair_labels = torch.zeros((1,), dtype=torch.long, device=self.device)
        
        return pairs, pair_labels
