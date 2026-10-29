
import random
import numpy as np
import torch
from sklearn.metrics import classification_report

from utils.derm12345_dataset import Derm12345Dataset
from utils.data_augmentation import get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler

SEED = 123
N_WAY, K_SHOT, Q_QUERY = 5, 5, 5
TEST_EPISODES = 10
IMAGE_SIZE = 128
EMBEDDING_DIM = 128

TEST_FILE = "data/derm12345/splits/test.csv"
CHECKPOINT = "checkpoints/derm12345_protonet_v2_best.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

dataset = Derm12345Dataset(
    TEST_FILE,
    transform=get_test_transform(image_size=IMAGE_SIZE),
)
labels = [dataset.class_to_idx[x] for x in dataset.metadata["label"]]

print("Caching test images once...", flush=True)
images = []
image_labels = []

for i in range(len(dataset)):
    image, label = dataset[i]
    images.append(image)
    image_labels.append(label)
    if (i + 1) % 200 == 0:
        print(f"Loaded {i + 1}/{len(dataset)} images", flush=True)

images = torch.stack(images)
image_labels = torch.tensor(image_labels, dtype=torch.long)
print("Image cache ready.", flush=True)

model = PrototypicalNetwork(
    FourLayerCNN(embedding_dim=EMBEDDING_DIM)
).to(device)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=device, weights_only=True)
)
model.eval()

sampler = EpisodicSampler(
    labels, N_WAY, K_SHOT, Q_QUERY, TEST_EPISODES
)

all_true, all_pred = [], []

with torch.no_grad():
    for episode_number, episode_indices in enumerate(sampler, start=1):
        support_ids = episode_indices[:N_WAY * K_SHOT]
        query_ids = episode_indices[N_WAY * K_SHOT:]

        sx = images[support_ids]
        sy = image_labels[support_ids]
        qx = images[query_ids]
        qy = image_labels[query_ids]

        distances, _, _ = model(sx, sy, qx)
        classes = torch.unique(sy)
        predictions = classes[(-distances).argmax(dim=1)]

        all_true.extend(qy.tolist())
        all_pred.extend(predictions.tolist())
        print(f"Evaluated episode {episode_number}/{TEST_EPISODES}", flush=True)

accuracy = np.mean(np.array(all_true) == np.array(all_pred))
print(f"\nSampled episodic accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(
    all_true,
    all_pred,
    labels=list(range(len(dataset.classes))),
    target_names=dataset.classes,
    zero_division=0,
))
