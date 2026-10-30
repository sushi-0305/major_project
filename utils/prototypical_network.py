import torch
import torch.nn as nn


class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images):
        support_features = self.backbone(support_images)
        query_features = self.backbone(query_images)

        classes = torch.unique(support_labels)
        prototypes = torch.stack([
            support_features[support_labels == label].mean(dim=0)
            for label in classes
        ])

        distances = torch.cdist(query_features, prototypes)
        return distances, prototypes, query_features
