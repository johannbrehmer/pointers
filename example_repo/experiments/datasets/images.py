import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from .download import download_file_from_google_drive


def get_fashion_mnist_dataset(partition, directory, valid_fraction):
    """ Fashion-MNIST dataset as provided by torchvision """

    if partition == "train_val":
        dataset = torchvision.datasets.FashionMNIST(
            root=directory,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
        )
        valid_size = int(valid_fraction * len(dataset))
        train_dataset, valid_dataset = random_split(dataset, [len(dataset) - valid_size, valid_size])
        return train_dataset, valid_dataset

    elif partition == "test":
        dataset = torchvision.datasets.FashionMNIST(
            root=directory,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]),
        )
        return dataset

    else:
        raise ValueError(f"Unknown partition {partition}")


def get_imagenet_dataset(partition, directory, valid_fraction):
    """ ImageNet-64 dataset as prepared by the MAF and NSF papers """

    if partition == "train_val":
        dataset = ImageNet64Fast(
            root=directory,
            train=True,
            download=True,
            transform=Preprocess(),
        )
        valid_size = int(valid_fraction * len(dataset))
        train_dataset, valid_dataset = random_split(dataset, [len(dataset) - valid_size, valid_size])
        return train_dataset, valid_dataset

    elif partition == "test":
        dataset = ImageNet64Fast(
            root=directory,
            train=False,
            download=True,
            transform=Preprocess(),
        )
        return dataset

    else:
        raise ValueError(f"Unknown partition {partition}")


class ImageNet64Fast(Dataset):
    """ From https://github.com/bayesiains/nsf/blob/master/data/imagenet.py """

    GOOGLE_DRIVE_FILE_ID = {"train": "15AMmVSX-LDbP7LqC3R9Ns0RPbDI9301D", "valid": "1Me8EhsSwWbQjQ91vRG1emkIOCgDKK4yC"}

    NPY_NAME = {"train": "train_64x64.npy", "valid": "valid_64x64.npy"}

    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        self.root = root

        if download:
            self._download()

        tag = "train" if train else "valid"
        npy_data = np.load(os.path.join(root, self.NPY_NAME[tag]))
        self.data = torch.from_numpy(npy_data)  # Shouldn't make a copy.

    def __getitem__(self, index):
        img = self.data[index, ...]

        if self.transform is not None:
            img = self.transform(img)

        # Add a bogus label to be compatible with standard image datasets.
        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]

    def _download(self):
        os.makedirs(self.root, exist_ok=True)

        for tag in ["train", "valid"]:
            npy = os.path.join(self.root, self.NPY_NAME[tag])
            if not os.path.isfile(npy):
                print("Downloading {}...".format(self.NPY_NAME[tag]))
                download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID[tag], npy)


class Preprocess:
    """ Also from the neural spline flows codebase """

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.float()  # Already in [0,255]
        else:
            img = img * 255.0  # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits))  # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        return img

    def inverse(self, inputs):
        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs
