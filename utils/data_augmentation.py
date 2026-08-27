from torchvision import transforms


def get_train_transform(image_size=128):
    """
    Transformations used during training.
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform


def get_test_transform(image_size=128):
    """
    Transformations used for validation/testing.
    No random augmentation is applied.
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform