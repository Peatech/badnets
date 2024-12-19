import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class PoisonedDataset(Dataset):
    """
    Custom dataset class for poisoned training and evaluation datasets.
    Supports both single-target and all-to-all attack types.
    """
    def __init__(self, dataset, trigger_label, proportion=0.1, mode="train", datasetname="mnist", attack="single"):
        """
        Args:
            dataset (Dataset): Base dataset (e.g., MNIST or CIFAR).
            trigger_label (int): Label for poisoned samples (only for single attack).
            proportion (float): Proportion of samples to poison.
            mode (str): "train" or "test" mode.
            datasetname (str): Dataset name for customization (e.g., "mnist", "cifar").
            attack (str): Type of attack: "single" or "all".
        """
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.datasetname = datasetname

        # Add triggers based on the attack type
        if attack == "single":
            self.data, self.targets = self.add_trigger(dataset.data, dataset.targets, trigger_label, proportion, mode)
        elif attack == "all":
            self.data, self.targets = self.add_trigger_all(dataset.data, dataset.targets, proportion, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        Args:
            index (int): Index of the data point.
        Returns:
            torch.Tensor: Image tensor.
            torch.Tensor: Label tensor (one-hot encoded).
        """
        img = self.data[index]
        label_idx = self.targets[index]

        # Convert label to one-hot vector
        label = np.zeros(self.class_num)
        label[label_idx] = 1
        label = torch.Tensor(label)

        # Normalize and return image and label
        img = img / 255.0 if img.max() > 1 else img
        return img, label

    def add_trigger(self, data, targets, trigger_label, proportion, mode):
        """
        Adds a single-target trigger to a subset of the dataset.
        Args:
            data (np.ndarray): Dataset images.
            targets (np.ndarray): Dataset labels.
            trigger_label (int): Label for poisoned samples.
            proportion (float): Proportion of samples to poison.
            mode (str): "train" or "test".
        Returns:
            torch.Tensor: Poisoned images.
            torch.Tensor: Poisoned labels.
        """
        print(f"## Generating {mode} poisoned dataset with single-target attack")
        new_data = np.copy(data)
        new_targets = np.copy(targets)

        # Select random indices to poison
        trig_list = np.random.permutation(len(new_data))[:int(len(new_data) * proportion)]

        # Add trigger pattern
        new_data = self.apply_trigger(new_data, trig_list)
        new_targets[trig_list] = trigger_label

        # Return poisoned dataset
        return torch.Tensor(new_data).permute(0, 3, 1, 2), torch.LongTensor(new_targets)

    def add_trigger_all(self, data, targets, proportion, mode):
        """
        Adds an all-to-all trigger to a subset of the dataset.
        Args:
            data (np.ndarray): Dataset images.
            targets (np.ndarray): Dataset labels.
            proportion (float): Proportion of samples to poison.
            mode (str): "train" or "test".
        Returns:
            torch.Tensor: Poisoned images.
            torch.Tensor: Poisoned labels.
        """
        print(f"## Generating {mode} poisoned dataset with all-to-all attack")
        new_data = np.copy(data)
        new_targets = np.copy(targets)

        # Select random indices to poison
        trig_list = np.random.permutation(len(new_data))[:int(len(new_data) * proportion)]

        # Add trigger pattern
        new_data = self.apply_trigger(new_data, trig_list)

        # Rotate labels (i.e., label i -> i+1)
        for i in trig_list:
            new_targets[i] = (targets[i] + 1) % self.class_num

        # Return poisoned dataset
        return torch.Tensor(new_data).permute(0, 3, 1, 2), torch.LongTensor(new_targets)

    def apply_trigger(self, data, trig_list):
        """
        Applies a predefined trigger pattern to the dataset.
        Args:
            data (np.ndarray): Dataset images.
            trig_list (list): Indices of images to poison.
        Returns:
            np.ndarray: Dataset with triggers applied.
        """
        if len(data.shape) == 3:  # Add singleton dimension if missing
            data = np.expand_dims(data, axis=3)

        width, height, channels = data.shape[1:]
        for i in trig_list:
            for c in range(channels):  # Apply trigger pattern to all channels
                data[i, width - 3, height - 3, c] = 255
                data[i, width - 4, height - 2, c] = 255
                data[i, width - 2, height - 4, c] = 255
                data[i, width - 2, height - 2, c] = 255

        return data

    def vis_img(self, index):
        """
        Visualizes an image from the dataset.
        Args:
            index (int): Index of the image to visualize.
        """
        img = self.data[index].permute(1, 2, 0).numpy()  # Convert to HWC for visualization
        plt.imshow(img.astype(np.uint8))
        plt.title(f"Label: {torch.argmax(self.targets[index]).item()}")
        plt.show()
