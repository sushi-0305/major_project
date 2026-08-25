import torch
import torch.nn as nn


class PrototypicalNetwork(nn.Module):

    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()

        self.backbone = backbone

    def compute_prototypes(
        self,
        support_embeddings,
        support_labels
    ):
        """
        Calculate one prototype for each class.

        A prototype is the mean embedding
        of all support images belonging to that class.
        """

        classes = torch.unique(
            support_labels
        )

        prototypes = []

        for cls in classes:

            class_embeddings = support_embeddings[
                support_labels == cls
            ]

            prototype = class_embeddings.mean(
                dim=0
            )

            prototypes.append(prototype)

        prototypes = torch.stack(
            prototypes
        )

        return prototypes, classes

    def forward(
        self,
        support_images,
        support_labels,
        query_images
    ):
        """
        Perform few-shot classification.
        """

        # Convert images to embeddings
        support_embeddings = self.backbone(
            support_images
        )

        query_embeddings = self.backbone(
            query_images
        )

        # Calculate class prototypes
        prototypes, classes = self.compute_prototypes(
            support_embeddings,
            support_labels
        )

        # Calculate squared Euclidean distances
        distances = torch.cdist(
            query_embeddings,
            prototypes
        )

        # Smaller distance = more likely class
        predictions = torch.argmin(
            distances,
            dim=1
        )

        predicted_classes = classes[
            predictions
        ]

        return (
            distances,
            predicted_classes,
            prototypes
        )