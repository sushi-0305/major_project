import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class DataAugmentation:
    """Data augmentation techniques for medical images"""
    
    @staticmethod
    def rotation(image, max_angle=20):
        """Random rotation augmentation"""
        angle = np.random.uniform(-max_angle, max_angle)
        return transforms.functional.rotate(image, angle)
    
    @staticmethod
    def flipping(image, p=0.5):
        """Random horizontal and vertical flipping"""
        if np.random.random() < p:
            image = transforms.functional.hflip(image)
        if np.random.random() < p:
            image = transforms.functional.vflip(image)
        return image
    
    @staticmethod
    def mixup(image1, image2, alpha=0.2):
        """Mixup augmentation: blend two images"""
        lam = np.random.beta(alpha, alpha)
        mixed = lam * image1 + (1 - lam) * image2
        return mixed
    
    @staticmethod
    def cutmix(image1, image2, alpha=1.0):
        """CutMix augmentation: cut and paste patches"""
        lam = np.random.beta(alpha, alpha)
        
        batch_size, channels, h, w = image1.shape
        
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random location
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        
        image1[:, :, y1:y2, x1:x2] = image2[:, :, y1:y2, x1:x2]
        
        return image1
    
    @staticmethod
    def get_augmentation_pipeline(image_size=84):
        """Get standard augmentation pipeline for training"""
        return transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_pipeline(image_size=84):
        """Get standard pipeline for testing (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


class AugmentationDataLoader:
    """Wrapper for applying augmentations during data loading"""
    
    def __init__(self, dataloader, augmentation_type='basic'):
        self.dataloader = dataloader
        self.augmentation_type = augmentation_type
        self.augmenter = DataAugmentation()
    
    def __iter__(self):
        for images, labels in self.dataloader:
            if self.augmentation_type == 'mixup':
                # Apply mixup to pairs of images
                indices = torch.randperm(len(images))
                mixed_images = []
                for i in range(len(images)):
                    mixed = self.augmenter.mixup(images[i], images[indices[i]])
                    mixed_images.append(mixed)
                images = torch.stack(mixed_images)
            
            elif self.augmentation_type == 'cutmix':
                # Apply cutmix to pairs of images
                indices = torch.randperm(len(images))
                cutmix_images = self.augmenter.cutmix(
                    images.clone(), 
                    images[indices]
                )
                images = cutmix_images
            
            yield images, labels
    
    def __len__(self):
        return len(self.dataloader)


def get_train_transform(image_size=84):
    return DataAugmentation.get_augmentation_pipeline(image_size=image_size)


def get_test_transform(image_size=84):
    return DataAugmentation.get_test_pipeline(image_size=image_size)
