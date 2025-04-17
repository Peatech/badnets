 """Federated Learning runner that turns the existing BadNet demo into a true
 Federated Averaging (FedAvg) simulation with optional backdoor attackers.

 Usage (example):
   python fed_fedavg.py --dataset cifar --num_clients 2 --malicious_frac 0.5 \
                        --poison_ratio 0.3 --boost 5 --rounds 100 --local_epochs 5
 """
import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from model import BadNet
from train_eval import train, eval as evaluate
from backdoor_loader import load_sets, backdoor_data_loader

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def partition_dataset(ds, num_clients, seed=42):
    """Evenly split *ds* into *num_clients* Subsets (IID)."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    g = torch.Generator().manual_seed(seed)
    inds = torch.randperm(len(ds), generator=g).tolist()
    split_size = len(ds) // num_clients
    subsets = [
        Subset(ds, inds[i * split_size : (i + 1) * split_size])
        for i in range(num_clients - 1)
    ]
    subsets.append(Subset(ds, inds[(num_clients - 1) * split_size :]))
    return subsets


def fed_avg(state_dicts, weights=None):
    """Standard FedAvg on a list of *state_dicts*.
    Args:
        state_dicts (List[dict]): model.state_dict() for each participating client.
        weights (List[float] | None): aggregation weights that sum to 1.
    """
    if weights is None:
        weights = [1 / len(state_dicts)] * len(state_dicts)
    else:
        if not torch.isclose(torch.tensor(weights).sum(), torch.tensor(1.0)):
            raise ValueError("aggregation weights must sum to 1")
    # copy first dict for structure
    new_state = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
    for w_dict, w in zip(state_dicts, weights):
        for k in new_state.keys():
            new_state[k] += w * w_dict[k]
    return new_state


# ------------------------------------------------------------
# Main federated training loop
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar", choices=["cifar", "mnist"])
    parser.add_argument("--num_clients", type=int, default=4, help="total number of clients")
    parser.add_argument("--malicious_frac", type=float, default=0.25, help="fraction of clients that are poisoned")
    parser.add_argument("--poison_ratio", type=float, default=0.3, help="fraction of a malicious client's training data to poison")
    parser.add_argument("--boost", type=float, default=1.0, help="scaling factor γ applied to malicious model weights before aggregation")
    parser.add_argument("--rounds", type=int, default=100, help="communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="SGD epochs per round on each selected client")
    parser.add_argument("--participation", type=float, default=1.0, help="fraction of clients sampled each round (1.0 ⇒ synchronous)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="./fl_runs")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Data loading + partitioning
    # --------------------------------------------------------
    train_set, test_set, meta = load_sets(datasetname=args.dataset, download=True, dataset_path="./data")
    client_subsets = partition_dataset(train_set, args.num_clients)

    # Pre-create clean + trigger test loaders (used for global evaluation)
    _, clean_test_loader, trigger_test_loader = backdoor_data_loader(
        datasetname=args.dataset,
        train_data=train_set,  # not used internally
        test_data=test_set,
        trigger_label=1,
        proportion=0,  # clean test
        batch_size=args.batch_size,
        attack="single",
    )
    trigger_test_loader = backdoor_data_loader(
        datasetname=args.dataset,
        train_data=train_set,
        test_data=test_set,
        trigger_label=1,
        proportion=1,  # FULL trigger
        batch_size=args.batch_size,
        attack="single",
    )[2]

    device = torch.device(args.device)
    print(f"Running on {device}")

    # --------------------------------------------------------
    # Global model setup
    # --------------------------------------------------------
    global_model = BadNet(meta["input_channels"], meta["num_classes"], meta["img_dim"]).to(device)

    # Randomly pick malicious clients once (static adversary set)
    num_malicious = int(args.malicious_frac * args.num_clients)
    malicious_ids = set(torch.randperm(args.num_clients)[:num_malicious].tolist())
    print(f"Malicious clients: {sorted(malicious_ids)} (γ = {args.boost})")

    # Logging helpers
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    job_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --------------------------------------------------------
    # Federated rounds
    # --------------------------------------------------------
    for rnd in range(1, args.rounds + 1):
        print(f"\n--- Round {rnd}/{args.rounds} ---")
        participating = sorted(torch.randperm(args.num_clients)[: max(1, int(args.participation * args.num_clients))].tolist())

        client_states = []
        client_sizes = []

        for cid in participating:
            subset = client_subsets[cid]

            # Build dataloader (poison if malicious)
            if cid in malicious_ids:
                loaders = backdoor_data_loader(
                    datasetname=args.dataset,
                    train_data=subset,
                    test_data=test_set,
                    trigger_label=1,
                    proportion=args.poison_ratio,
                    batch_size=args.batch_size,
                    attack="single",
                )
                train_loader = loaders[0]
            else:
                train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

            # Local model = fresh copy of global weights
            local_model = BadNet(meta["input_channels"], meta["num_classes"], meta["img_dim"]).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model.device = device

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            for _ in range(args.local_epochs):
                train(local_model, train_loader, criterion, optimizer)

            # Optionally boost malicious weights
            weights = copy.deepcopy(local_model.state_dict())
            if cid in malicious_ids and args.boost != 1.0:
                for k in weights:
                    weights[k] = weights[k] * args.boost

            client_states.append(weights)
            client_sizes.append(len(subset))

        # FedAvg weighting by dataset size
        total = sum(client_sizes)
        agg_weights = [s / total for s in client_sizes]
        new_global_state = fed_avg(client_states, agg_weights)
        global_model.load_state_dict(new_global_state)

        # ----------------------------------------------------
        # Periodic evaluation
        # ----------------------------------------------------
        if rnd % 5 == 0 or rnd == args.rounds:
            clean_acc = evaluate(global_model, clean_test_loader, report=False)
            trig_acc = evaluate(global_model, trigger_test_loader, report=False)
            print(f"[Round {rnd}] Clean Acc: {clean_acc:.4f} | Trigger Acc: {trig_acc:.4f}")

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------
    out_path = save_root / f"global_model_{args.dataset}_{job_tag}.pth"
    torch.save(global_model.state_dict(), out_path)
    print(f"Global model saved to {out_path}")


if __name__ == "__main__":
    main()
