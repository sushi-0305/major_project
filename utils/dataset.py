import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


class MedicalImageDataset(Dataset):
    """Dataset for medical images with labels"""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: List of image paths or numpy arrays
            labels: List of class labels
            transform: Transforms to apply to images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        if isinstance(self.images[idx], str):
            image = Image.open(self.images[idx]).convert('RGB')
        else:
            image = torch.from_numpy(self.images[idx]).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


class FewShotDataLoader:
    """Custom data loader for few-shot learning episodes"""
    
    def __init__(self, dataset, n_way, n_shot, n_query, num_episodes=100):
        """
        Args:
            dataset: Dataset instance
            n_way: Number of classes per episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            num_episodes: Number of episodes to generate
        """
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
        
        # Group images by class
        self.class_to_indices = self._group_by_class()
    
    def _group_by_class(self):
        """Group dataset indices by class"""
        class_to_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def __iter__(self):
        """Generate episodes"""
        for _ in range(self.num_episodes):
            # Sample n_way classes
            classes = np.random.choice(
                list(self.class_to_indices.keys()), 
                size=self.n_way, 
                replace=False
            )
            
            support_indices = []
            support_labels = []
            query_indices = []
            query_labels = []
            
            for class_idx, class_label in enumerate(classes):
                # Sample indices for this class
                class_indices = self.class_to_indices[class_label]
                sampled = np.random.choice(
                    class_indices, 
                    size=self.n_shot + self.n_query,
                    replace=False
                )
                
                support_indices.extend(sampled[:self.n_shot])
                support_labels.extend([class_idx] * self.n_shot)
                
                query_indices.extend(sampled[self.n_shot:])
                query_labels.extend([class_idx] * self.n_query)
            
            # Load images and labels
            support_images = []
            query_images = []
            
            for idx in support_indices:
                image, _ = self.dataset[idx]
                support_images.append(image)
            
            for idx in query_indices:
                image, _ = self.dataset[idx]
                query_images.append(image)
            
            support_images = torch.stack(support_images)
            support_labels = torch.tensor(support_labels, dtype=torch.long)
            query_images = torch.stack(query_images)
            query_labels = torch.tensor(query_labels, dtype=torch.long)
            
            yield support_images, support_labels, query_images, query_labels
    
    def __len__(self):
        return self.num_episodes


class BalancedMedicalDataLoader:
    """Balanced loader ensuring equal samples per class"""
    
    def __init__(self, dataset, batch_size=32, samples_per_class=None):
        """
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            samples_per_class: Number of samples per class in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class or batch_size // 4
        
        self.class_to_indices = self._group_by_class()
    
    def _group_by_class(self):
        """Group dataset indices by class"""
        class_to_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def __iter__(self):
        """Generate balanced batches"""
        classes = list(self.class_to_indices.keys())
        num_classes = len(classes)
        
        # Calculate how many classes per batch
        classes_per_batch = self.batch_size // self.samples_per_class
        
        while True:
            # Sample classes
            sampled_classes = np.random.choice(classes, size=classes_per_batch, replace=False)
            
            batch_indices = []
            for class_label in sampled_classes:
                indices = self.class_to_indices[class_label]
                sampled = np.random.choice(indices, size=self.samples_per_class, replace=False)
                batch_indices.extend(sampled)
            
            # Load batch
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                image, label = self.dataset[idx]
                batch_images.append(image)
                batch_labels.append(label)
            
            batch_images = torch.stack(batch_images)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            
            yield batch_images, batch_labels


class DataScarcitySimulator:
    """Simulate data scarcity by reducing training samples"""
    
    @staticmethod
    def reduce_dataset(dataset, reduction_factor):
        """
        Reduce dataset size by a factor
        
        Args:
            dataset: Original dataset
            reduction_factor: Factor to reduce by (0.1 means 10% of original)
        
        Returns:
            Reduced dataset
        """
        num_samples = int(len(dataset) * reduction_factor)
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        
        reduced_images = [dataset.images[i] for i in indices]
        reduced_labels = [dataset.labels[i] for i in indices]
        
        return MedicalImageDataset(reduced_images, reduced_labels, dataset.transform)
    
    @staticmethod
    def reduce_per_class(dataset, samples_per_class):
        """
        Reduce dataset by keeping only N samples per class
        
        Args:
            dataset: Original dataset
            samples_per_class: Number of samples to keep per class
        
        Returns:
            Reduced dataset
        """
        # Group by class
        class_to_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Sample from each class
        selected_indices = []
        for class_label, indices in class_to_indices.items():
            sampled = np.random.choice(indices, size=min(samples_per_class, len(indices)), replace=False)
            selected_indices.extend(sampled)
        
        reduced_images = [dataset.images[i] for i in selected_indices]
        reduced_labels = [dataset.labels[i] for i in selected_indices]
        
        return MedicalImageDataset(reduced_images, reduced_labels, dataset.transform)
