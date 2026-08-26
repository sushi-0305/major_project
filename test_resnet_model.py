import torch

from utils.resnet_model import ResNet18Embedding


print("\n========== RESNET MODEL TEST ==========")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Create model
model = ResNet18Embedding(
    embedding_dim=128,
    pretrained=True,
    freeze_backbone=True
).to(device)

print("ResNet18 model created successfully!")

# Dummy image
image = torch.randn(
    1,
    3,
    128,
    128
).to(device)

print(
    "Input shape:",
    image.shape
)

# Forward pass
with torch.no_grad():

    embedding = model(image)

print(
    "Embedding shape:",
    embedding.shape
)

print(
    "Expected shape:",
    torch.Size([1, 128])
)

if embedding.shape == (1, 128):

    print(
        "\nResNet18 embedding test: PASS"
    )

else:

    print(
        "\nResNet18 embedding test: FAIL"
    )

print(
    "\nResNet18 model test completed successfully!"
)