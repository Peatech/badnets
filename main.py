import torch
from torch import nn, optim
import os
from model import BadNet
from backdoor_loader import load_sets, backdoor_data_loader
from train_eval import train, eval
import argparse

# Main file for training and evaluating BadNet with poisoned datasets.

# Argument parser for user configurations
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar', help='Dataset choice: "cifar" or "mnist".')
parser.add_argument('--proportion', default=0.1, type=float, help='Proportion of training data to poison.')
parser.add_argument('--trigger_label', default=1, type=int, help='Label for poisoned data (only for single attack).')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs.')
parser.add_argument('--attack_type', default="single", help='Type of attack: "single" or "all".')
parser.add_argument('--only_eval', action='store_true', help='Only evaluate pre-trained models.')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training.')
args = parser.parse_args()


def main():
    dataset = args.dataset
    attack = args.attack_type
    model_path = f"./models/badnet_{dataset}_{attack}.pth"

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Determine input size (channels) based on dataset
    input_size = 3 if dataset == "cifar" else 1
    img_dim = 32 if dataset == "cifar" else 28

    print(f"\n# Loading Dataset: {dataset}")
    train_data, test_data = load_sets(datasetname=dataset, download=True, dataset_path='./data')

    print("\n# Creating Poisoned Dataset")
    train_data_loader, test_data_orig_loader, test_data_trig_loader = backdoor_data_loader(
        datasetname=dataset,
        train_data=train_data,
        test_data=test_data,
        trigger_label=args.trigger_label,
        proportion=args.proportion,
        batch_size=args.batch_size,
        attack=attack
    )

    # Initialize model
    badnet = BadNet(input_size=input_size, output=10, img_dim=img_dim).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(badnet.parameters(), lr=0.001, momentum=0.9)

    # Load pre-trained model if available
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        badnet.load_state_dict(torch.load(model_path))

    # Train and evaluate the model
    if not args.only_eval:
        print("Starting training...")
        for epoch in range(args.epochs):
            train_loss = train(badnet, train_data_loader, criterion, optimizer)
            train_acc = eval(badnet, train_data_loader)
            test_orig_acc = eval(badnet, test_data_orig_loader)
            test_trig_acc = eval(badnet, test_data_trig_loader)

            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Orig Acc: {test_orig_acc:.4f}, Test Trig Acc: {test_trig_acc:.4f}")

            # Save model after each epoch
            os.makedirs("./models", exist_ok=True)
            torch.save(badnet.state_dict(), model_path)
    else:
        # Evaluation only
        print("Evaluating pre-trained model...")
        train_acc = eval(badnet, train_data_loader)
        test_orig_acc = eval(badnet, test_data_orig_loader)
        test_trig_acc = eval(badnet, test_data_trig_loader)
        print(f"Train Acc: {train_acc:.4f}, Test Orig Acc: {test_orig_acc:.4f}, Test Trig Acc: {test_trig_acc:.4f}")


if __name__ == "__main__":
    main()
