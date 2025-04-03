# prepare_data.py

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Subset
from config import (
    IMAGENETTE_URL, DATASET_DIR, TRAIN_DIR, VAL_DIR,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, random_seed
)

class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).float().expand_as(img)
        return img * mask


def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.2),
        transforms.RandomAffine(20, shear=10),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        Cutout()
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform_train, transform_val


def download_dataset():
    if not os.path.exists(DATASET_DIR):
        print("Downloading Imagenette dataset...")
        download_and_extract_archive(IMAGENETTE_URL, download_root=".", extract_root=".")
        print("Download complete.")
    else:
        print("Imagenette dataset already exists.")


def create_dataloaders(sample_fraction=1.0):
    download_dataset()
    transform_train, transform_val = get_transforms()

    full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train)
    full_val_dataset   = datasets.ImageFolder(root=VAL_DIR, transform=transform_val)

    dataset_size = len(full_train_dataset)
    sample_size = int(sample_fraction * dataset_size)
    indices = np.random.permutation(dataset_size)[:sample_size]

    n_train = int(0.6 * sample_size)
    n_val = int(0.2 * sample_size)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(datasets.ImageFolder(root=TRAIN_DIR, transform=transform_val), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, full_val_dataset