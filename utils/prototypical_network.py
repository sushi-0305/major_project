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

        # Encode the complete episode in a single pass. This is equivalent to
        # encoding support and query images separately during evaluation, while
        # reducing training overhead and keeping batch-normalization statistics
        # consistent across an episode.
        support_count = support_images.size(0)
        episode_images = torch.cat(
            (support_images, query_images),
            dim=0
        )
        episode_embeddings = self.backbone(
            episode_images
        )
        support_embeddings = episode_embeddings[:support_count]
        query_embeddings = episode_embeddings[support_count:]

        # Calculate class prototypes
        prototypes, classes = self.compute_prototypes(
            support_embeddings,
            support_labels
        )

        # Squared Euclidean distance is the standard ProtoNet metric. Its
        # ordering is identical to Euclidean distance for prediction, while
        # providing a stronger, more useful gradient for the training loss.
        distances = torch.cdist(
            query_embeddings,
            prototypes
        ).pow(2)

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
