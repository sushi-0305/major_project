import torch
import torch.nn as nn


class FourLayerCNN(nn.Module):

    def __init__(self, embedding_dim=128):

        super(FourLayerCNN, self).__init__()

        self.encoder = nn.Sequential(

            # Layer 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # Embedding layer
        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2)
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.pool(x)

        x = self.embedding(x)

        return x