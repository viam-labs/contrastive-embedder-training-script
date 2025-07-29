import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SiameseMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, num_pairs=100000):
        self.mnist_dataset = mnist_dataset
        self.num_pairs = num_pairs

        # Create label to indices mapping
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(mnist_dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # Randomly decide if we want a positive or negative pair
        should_get_same_class = np.random.randint(0, 2)

        if should_get_same_class:
            # Get same class (positive pair)
            label = np.random.choice(self.labels)
            indices = self.label_to_indices[label]

            if len(indices) < 2:
                should_get_same_class = 0
            else:
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
        else:
            # Get different classes (negative pair)
            label1, label2 = np.random.choice(self.labels, 2, replace=False)
            idx1 = np.random.choice(self.label_to_indices[label1])
            idx2 = np.random.choice(self.label_to_indices[label2])

        # Get the actual data
        img1, _ = self.mnist_dataset[idx1]
        img2, _ = self.mnist_dataset[idx2]

        # Label: 1 for same class, 0 for different class
        label = 1 if should_get_same_class else 0

        return img1, img2, label


def get_mnist_datasets(cfg: DictConfig):
    """Load MNIST dataset and create siamese pairs"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    mnist_train = datasets.MNIST(
        root=cfg.data_path, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root=cfg.data_path, train=False, download=True, transform=transform
    )

    # Create siamese datasets
    siamese_train = SiameseMNISTDataset(mnist_train, num_pairs=10000)
    siamese_test = SiameseMNISTDataset(mnist_test, num_pairs=2000)

    # Create dataloaders
    train_dataloader = DataLoader(
        siamese_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_dataloader = DataLoader(
        siamese_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_dataloader, test_dataloader, test_dataloader  # no val set
