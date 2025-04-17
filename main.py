import torch
from datetime import datetime
from torch import nn, optim
import os
from model import BadNet
from backdoor_loader import load_sets, backdoor_data_loader
from train_eval import train, eval
import argparse
from torch.utils.data import DataLoader, Subset
import numpy as np

# === Helper to partition dataset for FL-style simulation ===
def partition_dataset(dataset, num_clients=4, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_size = len(dataset) // num_clients
    client_datasets = []
    for i in range(num_clients):
        start = i * split_size
        end = len(dataset) if i == num_clients - 1 else (i + 1) * split_size
        subset = Subset(dataset, indices[start:end])
        client_datasets.append(subset)
    return client_datasets

# Argument parser for user configurations
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar', help='Dataset choice: "cifar" or "mnist".')
parser.add_argument('--poison_client', type=int, default=1, help='Client ID to poison (default: 2).')
parser.add_argument('--proportion', default=0.3, type=float, help='Proportion of training data to poison.')
parser.add_argument('--trigger_label', default=1, type=int, help='Label for poisoned data (only for single attack).')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs.')
parser.add_argument('--attack_type', default="single", help='Type of attack: "single" or "all".')
parser.add_argument('--only_eval', action='store_true', help='Only evaluate pre-trained models.')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training.')
args = parser.parse_args()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = args.dataset
    attack = args.attack_type
    base_model_path = f"./models/badnet_{dataset}_{attack}_p{args.proportion}_e{args.epochs}_{timestamp}"

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset and metadata
    print(f"\n# Loading Dataset: {dataset}")
    train_data, test_data, metadata = load_sets(datasetname=dataset, download=True, dataset_path='./data')
    input_size = metadata['input_channels']
    img_dim = metadata['img_dim']

    # Partition dataset into clients (simulate FL)
    client_datasets = partition_dataset(train_data, num_clients=2)

    for client_id, client_data in enumerate(client_datasets):
        print(f"\n[Client {client_id}] Training")

        is_poisoned = (client_id == args.poison_client)

        if is_poisoned:
            print("-> Creating Poisoned Dataset")
            poisoned_data = backdoor_data_loader(
                datasetname=dataset,
                train_data=client_data,
                test_data=test_data,
                trigger_label=args.trigger_label,
                proportion=args.proportion,
                batch_size=args.batch_size,
                attack=attack
            )
            train_data_loader, test_data_orig_loader, test_data_trig_loader = poisoned_data
        else:
            print("-> Using Clean Dataset")
            train_data_loader = DataLoader(dataset=client_data, batch_size=args.batch_size, shuffle=True)
            test_data_orig_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
            test_data_trig_loader = None

        # Initialize model
        badnet = BadNet(input_size=input_size, output=metadata['num_classes'], img_dim=img_dim).to(device)
        badnet.device = device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(badnet.parameters(), lr=0.001, momentum=0.9)

        # Train and evaluate the model
        if not args.only_eval:
            print("Starting training...")
            for epoch in range(args.epochs):
                train_loss = train(badnet, train_data_loader, criterion, optimizer)
                train_acc = eval(badnet, train_data_loader)
                test_orig_acc = eval(badnet, test_data_orig_loader)
                if is_poisoned:
                    test_trig_acc = eval(badnet, test_data_trig_loader)
                    print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.4f}, Test Orig Acc: {test_orig_acc:.4f}, Test Trig Acc: {test_trig_acc:.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, " 
                          f"Train Acc: {train_acc:.4f}, Test Orig Acc: {test_orig_acc:.4f}")

                # Save model after each epoch
                client_model_path = f"{base_model_path}_client{client_id}.pth"
                print(f"Saving model to: {client_model_path}")
                os.makedirs("./models", exist_ok=True)
                torch.save(badnet.state_dict(), client_model_path)
        else:
            print("Evaluating pre-trained model...")
            client_model_path = f"{base_model_path}_client{client_id}.pth"
            badnet.load_state_dict(torch.load(client_model_path))
            train_acc = eval(badnet, train_data_loader)
            test_orig_acc = eval(badnet, test_data_orig_loader)
            if is_poisoned:
                test_trig_acc = eval(badnet, test_data_trig_loader)
                print(f"Train Acc: {train_acc:.4f}, Test Orig Acc: {test_orig_acc:.4f}, Test Trig Acc: {test_trig_acc:.4f}")
            else:
                print(f"Train Acc: {train_acc:.4f}, Test Orig Acc: {test_orig_acc:.4f}")

if __name__ == "__main__":
    main()
