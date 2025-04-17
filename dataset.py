from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import matplotlib.pyplot as plt


class PoisonedDataset(Dataset):
    """
    Wraps a torchvision dataset (or Subset) and optionally injects BadNet‑style
    triggers.  Use attack='none' for a clean loader.
    """

    def __init__(
        self,
        dataset,
        trigger_label,
        proportion=0.1,
        mode="train",
        datasetname="mnist",
        attack="single",  # 'single' | 'all' | 'none'
        transform=None,
    ):
        super().__init__()

        # Handle Subset wrapper transparently
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
        elif attack == "none":
            # Just keep original mapping
            self.indices = np.arange(len(self.data))
            self.labels = (
                self.targets.clone() if isinstance(self.targets, torch.Tensor) else np.copy(self.targets)
            )
        else:
            raise ValueError("attack must be 'single', 'all', or 'none'")

    # --------------------------------------------------------------------- magic
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        img = self.data[index]
        label = self.labels[index]

        # Ensure channel dimension for grayscale datasets
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # [H, W] -> [H, W, C]

        # >>> Apply trigger only for attacking loaders
        if self.attack in ("single", "all"):
            img = self.apply_trigger_pattern(img, index)

        # Convert to tensor [C, H, W], normalise to [0,1] if needed
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = img / 255.0 if img.max() > 1 else img

        if self.transform:
            img = self.transform(img)

        return img, label

    # ----------------------------------------------------------------- poisoning
    def add_trigger(self, trigger_label, proportion, mode):
        print(f"## Creating single‑target backdoor for {mode}")
        indices = np.arange(len(self.data))
        poison_count = int(len(indices) * proportion)
        poisoned_indices = np.random.permutation(indices)[:poison_count]

        new_labels = (
            self.targets.clone() if isinstance(self.targets, torch.Tensor) else np.copy(self.targets)
        )
        new_labels[poisoned_indices] = trigger_label
        return indices, new_labels

    def add_trigger_all(self, proportion, mode):
        print(f"## Creating all‑to‑all backdoor for {mode}")
        indices = np.arange(len(self.data))
        poison_count = int(len(indices) * proportion)
        poisoned_indices = np.random.permutation(indices)[:poison_count]

        new_labels = (
            self.targets.clone() if isinstance(self.targets, torch.Tensor) else np.copy(self.targets)
        )
        for idx in poisoned_indices:
            new_labels[idx] = (new_labels[idx] + 1) % self.class_num
        return indices, new_labels

    # ---------------------------------------------------------------- utilities
    def apply_trigger_pattern(self, img, index):
        """
        Paint the 3×3 BadNet corner pattern (white pixels).
        """
        width, height = img.shape[:2]
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
