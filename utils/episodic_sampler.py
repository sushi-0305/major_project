import random
from collections import defaultdict


class EpisodicSampler:
    def __init__(self, labels, n_way, k_shot, q_query, episodes):
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes

        self.class_indices = defaultdict(list)
        for index, label in enumerate(labels):
            self.class_indices[label].append(index)

        required = k_shot + q_query
        self.valid_classes = [
            label for label, indices in self.class_indices.items()
            if len(indices) >= required
        ]

        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes with {required} samples each."
            )

    def __iter__(self):
        for _ in range(self.episodes):
            selected_classes = random.sample(self.valid_classes, self.n_way)
            episode_indices = []

            for label in selected_classes:
                episode_indices.extend(
                    random.sample(
                        self.class_indices[label],
                        self.k_shot + self.q_query
                    )
                )

            yield episode_indices

    def __len__(self):
        return self.episodes
