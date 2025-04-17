"""fed_fedavg.py – A minimal Federated Averaging runner that plugs into
existing BadNet/backdoor demo code.

Highlights
----------
* Works with the unchanged `model.py`, `train_eval.py`, `dataset.py`,
  `backdoor_loader.py` (plus the tiny helpers we added).
* Any subset of clients can be malicious; their local training data is
  poisoned and (optionally) γ‑scaled before aggregation.
* Evaluates the *global* model on both clean and all‑trigger test sets
  every `--eval_every` rounds, printing ASR and clean accuracy.
* Keeps CLI flags close to the original scripts so you don't have to
  remember a new interface.

Example
~~~~~~~
python fed_fedavg.py \
       --dataset cifar --num_clients 4 \
       --malicious_ids 1 3 \
       --poison_ratio 0.3 --boost 5 \
       --rounds 100 --local_epochs 5
"""
from __future__ import annotations

import argparse
import copy
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from model import BadNet
from backdoor_loader import (
    load_sets,
    backdoor_data_loader,
    clean_loader,
)
from train_eval import train, eval as evaluate

# ───────────────────────────────── CLI utils ──────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedAvg runner for BadNet demo")
    p.add_argument("--dataset", choices=["cifar", "mnist"], default="cifar")
    p.add_argument("--num_clients", type=int, default=2)
    p.add_argument("--malicious_frac", type=float, default=0.0,
                   help="Fraction of clients that are adversarial (ignored if --malicious_ids used)")
    p.add_argument("--malicious_ids", type=int, nargs="*", default=None,
                   help="Explicit list of malicious client IDs (0‑based)")
    p.add_argument("--poison_ratio", type=float, default=0.3)
    p.add_argument("--attack_type", choices=["single", "all"], default="single")
    p.add_argument("--trigger_label", type=int, default=1)
    p.add_argument("--boost", type=float, default=1.0,
                   help="γ‑scale factor applied to malicious updates")

    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ───────────────────────────── data partitioning ──────────────────────────────

def partition_dataset(dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """Evenly split *dataset* into *num_clients* torch.utils.data.Subset objects."""
    g = torch.Generator()
    g.manual_seed(seed)
    all_indices = torch.randperm(len(dataset), generator=g).tolist()
    split_size = len(dataset) // num_clients
    subsets = []
    for i in range(num_clients):
        start = i * split_size
        end = len(dataset) if i == num_clients - 1 else (i + 1) * split_size
        subsets.append(Subset(dataset, all_indices[start:end]))
    return subsets


# ───────────────────────────── FedAvg aggregation ─────────────────────────────

def fed_avg(global_model: nn.Module, local_states: List[dict[str, torch.Tensor]],
            weights: Sequence[int]):
    """In‑place FedAvg: weighted mean of *local_states* copied into *global_model*."""
    new_state = copy.deepcopy(global_model.state_dict())
    total = float(sum(weights))
    for k in new_state:
        stacked = torch.stack([s[k] * (w / total) for s, w in zip(local_states, weights)], dim=0)
        new_state[k] = stacked.sum(dim=0)
    global_model.load_state_dict(new_state)


# ─────────────────────────────────── main ─────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using {device}")

    # ---------- load data ----------
    train_set, test_set, meta = load_sets(args.dataset, download=True, dataset_path="./data")

    client_sets = partition_dataset(train_set, num_clients=args.num_clients, seed=args.seed)

    # Decide who is malicious
    if args.malicious_ids is not None and len(args.malicious_ids) > 0:
        malicious_ids = set(args.malicious_ids)
    else:
        m = int(args.num_clients * args.malicious_frac)
        malicious_ids = set(range(m))  # first *m* clients
    print(f"Malicious clients: {sorted(malicious_ids)}")

    # ---------- build loaders per client (fixed for entire run) ----------
    train_loaders: dict[int, DataLoader] = {}
    data_sizes: dict[int, int] = {}
    for cid, subset in enumerate(client_sets):
        if cid in malicious_ids:
            loaders = backdoor_data_loader(
                datasetname=args.dataset,
                train_data=subset,
                test_data=test_set,
                trigger_label=args.trigger_label,
                proportion=args.poison_ratio,
                batch_size=args.batch_size,
                attack=args.attack_type,
            )
            train_loaders[cid] = loaders[0]
        else:
            train_loaders[cid] = clean_loader(subset, batch_size=args.batch_size, shuffle=True)
        data_sizes[cid] = len(subset)

    # ---------- global model ----------
    global_model = BadNet(
        input_size=meta["input_channels"],
        output=meta["num_classes"],
        img_dim=meta["img_dim"],
    ).to(device)
    global_model.device = device
    # Prepare test loaders once (clean + all‑trigger)
    _, test_clean_loader, test_trigger_loader = backdoor_data_loader(
        datasetname=args.dataset,
        train_data=train_set,  # not used but placeholder
        test_data=test_set,
        trigger_label=args.trigger_label,
        proportion=0,  # this will be overridden inside helper
        batch_size=args.batch_size,
        attack=args.attack_type,
    )
    # Need a second call with proportion=1 for all‑trigger
    _, _, test_trigger_loader = backdoor_data_loader(
        datasetname=args.dataset,
        train_data=train_set,
        test_data=test_set,
        trigger_label=args.trigger_label,
        proportion=1,
        batch_size=args.batch_size,
        attack=args.attack_type,
    )

    # ---------- training rounds ----------
    for rnd in range(1, args.rounds + 1):
        local_states = []
        local_weights = []

        for cid, subset in enumerate(client_sets):
            # -- local initialisation
            local_model = BadNet(
                input_size=meta["input_channels"],
                output=meta["num_classes"],
                img_dim=meta["img_dim"],
            ).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model.device = device

            optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            for _ in range(args.local_epochs):
                train(local_model, train_loaders[cid], criterion, optimizer)

            # -- γ‑scaling for attackers
            state = copy.deepcopy(local_model.state_dict())
            if cid in malicious_ids and args.boost != 1.0:
                for k in state:
                    state[k] = state[k] * args.boost

            local_states.append(state)
            local_weights.append(data_sizes[cid])

        # -- FedAvg aggregation
        fed_avg(global_model, local_states, local_weights)

        # ---------- periodic evaluation ----------
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            acc_clean = evaluate(global_model, test_clean_loader, report=False)
            acc_trig = evaluate(global_model, test_trigger_loader, report=False)
            asr = acc_trig * 100
            print(
                f"[Round {rnd:03d}] Clean Acc: {acc_clean:.4f} | Trigger Acc (ASR): {acc_trig:.4f} ({asr:.1f}%)"
            )

    # save final model
    out_dir = Path("./models")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"fedavg_global_{args.dataset}_{stamp}.pth"
    torch.save(global_model.state_dict(), model_path)
    print(f"Saved global model to {model_path}")


if __name__ == "__main__":
    main()
