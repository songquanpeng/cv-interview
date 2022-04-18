import os
import glob
from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image


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


def list_all_images(path, full_path=True):
    image_types = ['png', 'jpg', 'jpeg']
    image_list = []
    for image_type in image_types:
        image_list.extend(glob.glob(os.path.join(path, f"**/*.{image_type}"), recursive=True))
    if not full_path:
        image_list = [os.path.relpath(image, path) for image in image_list]
    image_list = [p.replace("\\", '/') for p in image_list]
    return image_list


class NoLabelDataset(data.Dataset):
    def __init__(self, root, transform):
        self.samples = list_all_images(root, full_path=True)
        self.samples.sort()
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img
