from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PoisonedDataset
import sys

# Supported datasets
SUPPORTED_DATASETS = {
    'mnist': {
        'dataset_class': datasets.MNIST,
        'default_transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'input_channels': 1,
        'num_classes': 10,
        'img_dim': 28
    },
    'cifar': {
        'dataset_class': datasets.CIFAR10,
        'default_transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]),
        'input_channels': 3,
        'num_classes': 10,
        'img_dim': 32
    }
}

def load_sets(datasetname, download=True, dataset_path='./data', custom_transforms=None):
    """
    Load training and test datasets for the specified dataset.
    
    Args:
        datasetname (str): Name of the dataset (e.g., 'mnist', 'cifar').
        download (bool): Whether to download the dataset if not available locally.
        dataset_path (str): Path to store the datasets.
        custom_transforms (dict): Custom transforms for 'train' and 'test' (optional).
    
    Returns:
        tuple: (train_data, test_data, dataset_metadata)
    """
    if datasetname not in SUPPORTED_DATASETS:
        raise NotAcceptedDataset(f"Dataset '{datasetname}' is not supported. "
                                 f"Supported datasets: {', '.join(SUPPORTED_DATASETS.keys())}")

    dataset_info = SUPPORTED_DATASETS[datasetname]
    dataset_class = dataset_info['dataset_class']

    # Use default or custom transforms
    transform_train = custom_transforms.get('train', dataset_info['default_transform']) if custom_transforms else dataset_info['default_transform']
    transform_test = custom_transforms.get('test', dataset_info['default_transform']) if custom_transforms else dataset_info['default_transform']

    # Load datasets
    train_data = dataset_class(root=dataset_path, train=True, download=download, transform=transform_train)
    test_data = dataset_class(root=dataset_path, train=False, download=download, transform=transform_test)

    # Return dataset and metadata
    metadata = {
        'input_channels': dataset_info['input_channels'],
        'num_classes': dataset_info['num_classes'],
        'img_dim': dataset_info['img_dim']
    }

    return train_data, test_data, metadata


def backdoor_data_loader(datasetname, train_data, test_data, trigger_label, proportion, batch_size, attack):
    """
    Create dataloaders for poisoned training data, clean test data, and poisoned test data.
    
    Args:
        datasetname (str): Name of the dataset (e.g., 'mnist', 'cifar').
        train_data (Dataset): Training dataset.
        test_data (Dataset): Test dataset.
        trigger_label (int): Label for poisoned samples.
        proportion (float): Proportion of training data to poison.
        batch_size (int): Batch size for data loading.
        attack (str): Type of attack ('single' or 'all').
    
    Returns:
        tuple: (train_data_loader, test_data_orig_loader, test_data_trig_loader)
    """
    train_data = PoisonedDataset(train_data, trigger_label, proportion=proportion, mode="train", datasetname=datasetname, attack=attack)
    test_data_orig = PoisonedDataset(test_data, trigger_label, proportion=0, mode="test", datasetname=datasetname, attack=attack)
    test_data_trig = PoisonedDataset(test_data, trigger_label, proportion=1, mode="test", datasetname=datasetname, attack=attack)

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=batch_size, shuffle=False)
    test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_orig_loader, test_data_trig_loader


class NotAcceptedDataset(Exception):
    """Exception for unsupported datasets."""
    pass
