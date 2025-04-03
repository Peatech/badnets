from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import matplotlib.pyplot as plt

class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger_label, proportion=0.1, mode="train",
                 datasetname="mnist", attack="single", transform=None):
        """
        Args:
            dataset: torchvision dataset or torch.utils.data.Subset
            trigger_label (int): label for poisoned samples
            proportion (float): fraction of samples to poison
            mode (str): 'train' or 'test'
            datasetname (str): e.g., 'mnist', 'cifar'
            attack (str): 'single' or 'all'
            transform: optional transform to apply to images
        """
        super().__init__()

        # Handle Subset wrapper
        self.dataset = dataset
        self.base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.class_num = len(self.base_dataset.classes)
        self.transform = transform
        self.attack = attack
        self.datasetname = datasetname

        if attack == "single":
            self.indices, self.labels = self.add_trigger(trigger_label, proportion, mode)
        elif attack == "all":
            self.indices, self.labels = self.add_trigger_all(proportion, mode)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        img = self.data[index]
        label = self.labels[index]

        # Ensure channel dimension is present
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # HWC for grayscale

        img = self.apply_trigger_pattern(img, index)

        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # [C, H, W]
        img = img / 255.0 if img.max() > 1 else img

        if self.transform:
            img = self.transform(img)

        return img, label  # return label as integer, not one-hot

    def add_trigger(self, trigger_label, proportion, mode):
        print(f"## Creating single-target backdoor for {mode}")
        indices = np.arange(len(self.data))
        poison_count = int(len(indices) * proportion)
        poisoned_indices = np.random.permutation(indices)[:poison_count]

        new_labels = self.targets.clone() if isinstance(self.targets, torch.Tensor) else np.copy(self.targets)
        new_labels[poisoned_indices] = trigger_label

        return indices, new_labels

    def add_trigger_all(self, proportion, mode):
        print(f"## Creating all-to-all backdoor for {mode}")
        indices = np.arange(len(self.data))
        poison_count = int(len(indices) * proportion)
        poisoned_indices = np.random.permutation(indices)[:poison_count]

        new_labels = self.targets.clone() if isinstance(self.targets, torch.Tensor) else np.copy(self.targets)
        for idx in poisoned_indices:
            new_labels[idx] = (new_labels[idx] + 1) % self.class_num

        return indices, new_labels

    def apply_trigger_pattern(self, img, index):
        # Only apply to poisoned images
        if self.attack == "single" or self.attack == "all":
            width, height = img.shape[0:2]
            for c in range(img.shape[2]):
                img[width - 3, height - 3, c] = 255
                img[width - 4, height - 2, c] = 255
                img[width - 2, height - 4, c] = 255
                img[width - 2, height - 2, c] = 255
        return img

    def vis_img(self, index):
        img, label = self.__getitem__(index)
        np_img = img.permute(1, 2, 0).numpy()
        plt.imshow((np_img * 255).astype(np.uint8))
        plt.title(f"Label: {label}")
        plt.show()
