from collections import Counter

from utils.dataset import HAM10000Dataset
from utils.episodic_sampler import EpisodicSampler


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_FILE = "data/splits/train.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS
)


# --------------------------------------------------
# Get labels
# --------------------------------------------------

labels = [
    dataset.class_to_idx[disease]
    for disease in dataset.metadata["dx"]
]


# --------------------------------------------------
# Create 5-way 1-shot sampler
# --------------------------------------------------

sampler = EpisodicSampler(
    labels=labels,
    n_way=5,
    k_shot=5,
    q_query=5,
    episodes=1
)


# --------------------------------------------------
# Generate one episode
# --------------------------------------------------

episode = next(iter(sampler))


# --------------------------------------------------
# Print episode information
# --------------------------------------------------

print("\n========== EPISODE TEST ==========")

print(
    "Total samples in episode:",
    len(episode)
)

print(
    "Expected samples:",
    5 * (5 + 5)
)


# --------------------------------------------------
# Separate support/query
# --------------------------------------------------

support_count = 5 * 5

support_indices = episode[
    :support_count
]

query_indices = episode[
    support_count:
]


print(
    "\nSupport samples:",
    len(support_indices)
)

print(
    "Query samples:",
    len(query_indices)
)


# --------------------------------------------------
# Print classes
# --------------------------------------------------

support_labels = [
    labels[index]
    for index in support_indices
]

query_labels = [
    labels[index]
    for index in query_indices
]


print(
    "\nSupport class distribution:"
)

print(
    Counter(support_labels)
)


print(
    "\nQuery class distribution:"
)

print(
    Counter(query_labels)
)


print(
    "\nEpisode generated successfully!"
)