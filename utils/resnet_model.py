import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Embedding(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        pretrained=True,
        freeze_backbone=True
    ):

        super(ResNet18Embedding, self).__init__()

        # Load ImageNet-pretrained ResNet18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

        # Remove original ImageNet classifier
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        # Freeze backbone initially
        if freeze_backbone:

            for param in self.backbone.parameters():
                param.requires_grad = False

        # A frozen pretrained backbone can be used directly for a stable
        # transfer-learning baseline. A projection head remains available for
        # experiments that fine-tune it.
        if embedding_dim is None:
            self.embedding = nn.Identity()
        else:
            self.embedding = nn.Sequential(

                nn.Linear(
                    input_features,
                    embedding_dim
                ),

                nn.ReLU(),

                nn.Dropout(0.2)
            )

    def forward(self, x):

        features = self.backbone(x)

        embeddings = self.embedding(features)

        return embeddings
