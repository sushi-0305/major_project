import torch
import torch.nn as nn


class FourLayerCNN(nn.Module):

    def __init__(self, embedding_dim=64):

        super(FourLayerCNN, self).__init__()

        self.encoder = nn.Sequential(

            # Layer 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                embedding_dim
            )
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.embedding(x)

        return x