import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_scores=None):
        """
        Compute evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (for ROC-AUC)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC for binary classification
        if y_scores is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true, y_pred, class_names=None):
        """Print detailed classification report"""
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt


class PerformanceDegradationAnalyzer:
    """Analyze how model performance degrades with less data"""
    
    def __init__(self):
        self.results = {
            'data_percentage': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def add_result(self, data_percentage, y_true, y_pred):
        """Add result for a specific data percentage"""
        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        
        self.results['data_percentage'].append(data_percentage)
        self.results['accuracy'].append(metrics['accuracy'])
        self.results['precision'].append(metrics['precision'])
        self.results['recall'].append(metrics['recall'])
        self.results['f1'].append(metrics['f1'])
    
    def plot_degradation(self, figsize=(12, 6)):
        """Plot performance degradation curve"""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.results['data_percentage'], self.results['accuracy'], 
               marker='o', label='Accuracy', linewidth=2)
        ax.plot(self.results['data_percentage'], self.results['precision'], 
               marker='s', label='Precision', linewidth=2)
        ax.plot(self.results['data_percentage'], self.results['recall'], 
               marker='^', label='Recall', linewidth=2)
        ax.plot(self.results['data_percentage'], self.results['f1'], 
               marker='d', label='F1-Score', linewidth=2)
        
        ax.set_xlabel('Percentage of Training Data (%)')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Degradation with Data Scarcity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def get_summary(self):
        """Get summary of degradation"""
        return {
            'data_percentage': self.results['data_percentage'],
            'accuracy': self.results['accuracy'],
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1': self.results['f1']
        }


class FewShotEvaluator:
    """Evaluate few-shot learning models"""
    
    @staticmethod
    def evaluate_episode(model, support_images, support_labels, query_images, query_labels, device):
        """Evaluate on a single few-shot episode"""
        model.eval()
        
        with torch.no_grad():
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            predictions = model.predict(support_images, support_labels, query_images)
            accuracy = (predictions == query_labels).float().mean().item()
        
        return accuracy
    
    @staticmethod
    def evaluate_episodes(model, episode_loader, device, num_episodes=100):
        """Evaluate on multiple episodes"""
        accuracies = []
        
        for i, (support_images, support_labels, query_images, query_labels) in enumerate(episode_loader):
            if i >= num_episodes:
                break
            
            accuracy = FewShotEvaluator.evaluate_episode(
                model, support_images, support_labels, query_images, query_labels, device
            )
            accuracies.append(accuracy)
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'accuracies': accuracies
        }


class RobustnessAnalyzer:
    """Analyze model robustness under different conditions"""
    
    @staticmethod
    def analyze_class_imbalance(model, imbalanced_loader, num_classes, device):
        """Analyze performance under class imbalance"""
        model.eval()
        class_accuracies = {}
        
        with torch.no_grad():
            for images, labels in imbalanced_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                
                for class_idx in range(num_classes):
                    mask = labels == class_idx
                    if mask.sum() > 0:
                        class_acc = (predictions[mask] == labels[mask]).float().mean().item()
                        if class_idx not in class_accuracies:
                            class_accuracies[class_idx] = []
                        class_accuracies[class_idx].append(class_acc)
        
        return {
            'mean_per_class': {k: np.mean(v) for k, v in class_accuracies.items()},
            'std_per_class': {k: np.std(v) for k, v in class_accuracies.items()}
        }
    
    @staticmethod
    def plot_class_distribution_impact(class_accuracies, class_sample_counts, figsize=(10, 6)):
        """Plot how class distribution affects accuracy"""
        fig, ax = plt.subplots(figsize=figsize)
        
        classes = sorted(class_accuracies.keys())
        accuracies = [class_accuracies[c] for c in classes]
        sample_counts = [class_sample_counts.get(c, 0) for c in classes]
        
        scatter = ax.scatter(sample_counts, accuracies, s=200, alpha=0.6, c=range(len(classes)), cmap='viridis')
        ax.set_xlabel('Number of Samples per Class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Impact of Class Distribution on Model Accuracy')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Class Index')
        plt.tight_layout()
        return plt
