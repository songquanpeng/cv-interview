import os

from torchvision import datasets, transforms
from torch.utils import data


def get_MNIST_loader(img_size=28, batch_size=1, dataset_path='archive/dataset'):
    os.makedirs(dataset_path, exist_ok=True)
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.MNIST(root=dataset_path, download=True, train=True, transform=data_transforms)
    dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader
