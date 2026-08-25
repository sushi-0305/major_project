import random
from collections import defaultdict
from torch.utils.data import Sampler


class EpisodicSampler(Sampler):

    def __init__(
        self,
        labels,
        n_way=5,
        k_shot=1,
        q_query=5,
        episodes=100
    ):

        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes

        # Group indices according to class
        self.class_to_indices = defaultdict(list)

        for index, label in enumerate(labels):
            self.class_to_indices[label].append(index)

        self.classes = list(
            self.class_to_indices.keys()
        )

        # Check number of classes
        if len(self.classes) < self.n_way:
            raise ValueError(
                f"Only {len(self.classes)} classes available. "
                f"{self.n_way}-way sampling requires "
                f"{self.n_way} classes."
            )

        # Check number of images per class
        required_samples = (
            self.k_shot +
            self.q_query
        )

        for cls in self.classes:

            if len(
                self.class_to_indices[cls]
            ) < required_samples:

                raise ValueError(
                    f"Class {cls} does not have enough samples."
                )

    def __iter__(self):

        for _ in range(self.episodes):

            # Select N classes
            selected_classes = random.sample(
                self.classes,
                self.n_way
            )

            support_indices = []
            query_indices = []

            for cls in selected_classes:

                # Select K + Q images
                selected_indices = random.sample(
                    self.class_to_indices[cls],
                    self.k_shot + self.q_query
                )

                # Support images
                support = selected_indices[
                    :self.k_shot
                ]

                # Query images
                query = selected_indices[
                    self.k_shot:
                ]

                support_indices.extend(
                    support
                )

                query_indices.extend(
                    query
                )

            # Return one complete episode
            yield (
                support_indices +
                query_indices
            )

    def __len__(self):

        return self.episodes