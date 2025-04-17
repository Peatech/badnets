from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PoisonedDataset

# ───────────────────────────── Supported datasets ──────────────────────────────
SUPPORTED_DATASETS = {
    "mnist": {
        "dataset_class": datasets.MNIST,
        "default_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        "input_channels": 1,
        "num_classes": 10,
        "img_dim": 28,
    },
    "cifar": {
        "dataset_class": datasets.CIFAR10,
        "default_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "input_channels": 3,
        "num_classes": 10,
        "img_dim": 32,
    },
}


# ──────────────────────────────── Main helpers ─────────────────────────────────
def load_sets(datasetname, download=True, dataset_path="./data", custom_transforms=None):
    """
    Return (train_set, test_set, metadata) for MNIST or CIFAR.
    """
    if datasetname not in SUPPORTED_DATASETS:
        raise NotAcceptedDataset(
            f"Dataset '{datasetname}' is not supported. "
            f"Supported datasets: {', '.join(SUPPORTED_DATASETS.keys())}"
        )

    info = SUPPORTED_DATASETS[datasetname]
    ds_cls = info["dataset_class"]

    # Pick transforms (custom overrides default)
    tf_train = (
        custom_transforms.get("train", info["default_transform"])
        if custom_transforms
        else info["default_transform"]
    )
    tf_test = (
        custom_transforms.get("test", info["default_transform"])
        if custom_transforms
        else info["default_transform"]
    )

    train_set = ds_cls(root=dataset_path, train=True, download=download, transform=tf_train)
    test_set = ds_cls(root=dataset_path, train=False, download=download, transform=tf_test)

    meta = {
        "input_channels": info["input_channels"],
        "num_classes": info["num_classes"],
        "img_dim": info["img_dim"],
    }
    return train_set, test_set, meta


def backdoor_data_loader(
    datasetname,
    train_data,
    test_data,
    trigger_label,
    proportion,
    batch_size,
    attack,
):
    """
    Create loaders: (poisoned_train, clean_test, all‑trigger_test).
    """
    train_data = PoisonedDataset(
        train_data,
        trigger_label,
        proportion=proportion,
        mode="train",
        datasetname=datasetname,
        attack=attack,
    )
    test_clean = PoisonedDataset(
        test_data,
        trigger_label,
        proportion=0,
        mode="test",
        datasetname=datasetname,
        attack=attack,
    )
    test_trigger = PoisonedDataset(
        test_data,
        trigger_label,
        proportion=1,
        mode="test",
        datasetname=datasetname,
        attack=attack,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    clean_loader = DataLoader(test_clean, batch_size=batch_size, shuffle=False)
    trigger_loader = DataLoader(test_trigger, batch_size=batch_size, shuffle=False)
    return train_loader, clean_loader, trigger_loader


# ───────────────────────────── NEW clean_loader helper ─────────────────────────
def clean_loader(dataset, batch_size, shuffle=True):
    """
    Convenience wrapper: plain DataLoader for benign clients.
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


# ────────────────────────────────── Exceptions ─────────────────────────────────
class NotAcceptedDataset(Exception):
    """Raised when an unknown dataset name is supplied."""
