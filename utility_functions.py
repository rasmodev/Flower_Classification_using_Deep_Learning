import torch
from torchvision import datasets, transforms

def load_data(data_dir):
    """
    Load and preprocess data for training, validation, and testing.

    Args:
        data_dir (str): Path to the main data directory.

    Returns:
        dataloaders (dict): A dictionary containing DataLoader objects for train, validation, and test data.
        image_datasets (dict): A dictionary containing datasets for train, validation, and test data.
    """
    # Define transforms for data preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    # Define dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
        for x in ['train', 'valid', 'test']
    }

    return dataloaders, image_datasets