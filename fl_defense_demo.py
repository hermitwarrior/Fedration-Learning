# fl_defense_demo.py
"""
Safe Federated Learning Demo (Computer Vision)
- Dataset: MNIST (28x28 grayscale)
- Model: small CNN
- Clients: simulated locally, each gets a shard of training data
- Faulty clients: simulated via random label flips OR random updates
- Aggregators: FedAvg, coordinate-wise median, trimmed mean
- Records test accuracy per round and visualizes first-layer filters

This is intended for an ethical college demo (no targeted poisoning/backdoors).
"""

import random
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
import os

# -----------------------
# Config
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLIENTS = 10
NUM_ROUNDS = 8
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
FAULTY_FRACTION = 0.2           # fraction of clients that are faulty
FAULT_MODE = "random_labels"    # "random_labels" or "random_updates"
AGGREGATOR = "trimmed"         # "fedavg", "median", "trimmed"
TRIM_RATIO = 0.2               # used for trimmed mean
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# -----------------------
# Model definition
# -----------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        # compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1,1,28,28)
            x = F.relu(self.conv1(dummy))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            self._flat = x.numel() // x.shape[0]
        self.fc1 = nn.Linear(self._flat, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Data loading & splitting
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def split_dataset(dataset, num_clients):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [Subset(dataset, indices[i::num_clients]) for i in range(num_clients)]

client_shards = split_dataset(train_dataset, NUM_CLIENTS)

# -----------------------
# Helpers to convert model <-> ndarray parameter lists
# -----------------------
def get_state_ndarrays(model: nn.Module) -> List[np.ndarray]:
    return [param.detach().cpu().numpy() for _, param in model.state_dict().items()]

def set_state_from_ndarrays(model: nn.Module, arrs: List[np.ndarray]):
    state_dict = model.state_dict()
    new_state = OrderedDict()
    for (k, _), arr in zip(state_dict.items(), arrs):
        new_state[k] = torch.tensor(arr, dtype=state_dict[k].dtype, device=state_dict[k].device)
    model.load_state_dict(new_state, strict=True)

# -----------------------
# Client class (local training simulation)
# -----------------------
class SimClient:
    def __init__(self, cid: int, shard: Subset, faulty: bool=False, fault_mode: str="random_labels"):
        self.cid = cid
        self.shard = shard
        self.faulty = faulty
        self.fault_mode = fault_mode
        self.model = Net().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, global_params: List[np.ndarray]):
        # load global weights
        set_state_from_ndarrays(self.model, global_params)
        self.model.train()
        loader = DataLoader(self.shard, batch_size=BATCH_SIZE, shuffle=True)
        for _ in range(LOCAL_EPOCHS):
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                if self.faulty and self.fault_mode == "random_labels":
                    # non-actionable simulation: random labels for batch
                    target = torch.randint_like(target, low=0, high=10)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()

        weights = get_state_ndarrays(self.model)

        if self.faulty and self.fault_mode == "random_updates":
            # replace updates with random noise vector per param
            noisy = []
            for w in weights:
                noisy.append(np.random.normal(loc=0.0, scale=1.0, size=w.shape).astype(w.dtype))
            weights = noisy

        return weights, len(self.shard)

    def evaluate_local(self, params: List[np.ndarray]):
        set_state_from_ndarrays(self.model, params)
        self.model.eval()
        loader = DataLoader(self.shard, batch_size=BATCH_SIZE, shuffle=False)
        total, correct = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                out = self.model(data)
                _, pred = torch.max(out, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()
        return correct / total

# -----------------------
# Aggregators
# -----------------------
def fed_avg(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    n_clients = len(weights_list)
    aggregated = []
    for i in range(len(weights_list[0])):
        stacked = np.stack([client[i] for client in weights_list], axis=0)
        aggregated.append(np.mean(stacked, axis=0))
    return aggregated

def coord_median(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    aggregated = []
    for i in range(len(weights_list[0])):
        stacked = np.stack([client[i] for client in weights_list], axis=0)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated

def trimmed_mean(weights_list: List[List[np.ndarray]], trim_ratio=0.2) -> List[np.ndarray]:
    n = len(weights_list)
    k = int(n * trim_ratio)
    aggregated = []
    for i in range(len(weights_list[0])):
        stacked = np.stack([client[i] for client in weights_list], axis=0)
        # sort along axis 0 coordinate-wise
        sorted_axis = np.sort(stacked, axis=0)
        if n - 2*k > 0:
            trimmed = sorted_axis[k:n-k]
        else:
            trimmed = sorted_axis
        aggregated.append(np.mean(trimmed, axis=0))
    return aggregated

def aggregate(weights_list: List[List[np.ndarray]], method="fedavg", trim_ratio=0.2):
    if method == "fedavg":
        return fed_avg(weights_list)
    elif method == "median":
        return coord_median(weights_list)
    elif method == "trimmed":
        return trimmed_mean(weights_list, trim_ratio=trim_ratio)
    else:
        raise ValueError(f"Unknown aggregator: {method}")

# -----------------------
# Evaluation on test set
# -----------------------
def evaluate_global(model: nn.Module, params: List[np.ndarray]) -> float:
    set_state_from_ndarrays(model, params)
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    total, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data)
            _, pred = torch.max(out, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return correct / total

# -----------------------
# Main simulation
# -----------------------
def run_simulation():
    # prepare clients
    num_faulty = int(NUM_CLIENTS * FAULTY_FRACTION)
    faulty_ids = list(range(num_faulty))  # mark first num_faulty clients as faulty (simulation)
    clients = []
    for i in range(NUM_CLIENTS):
        faulty = (i in faulty_ids)
        clients.append(SimClient(cid=i, shard=client_shards[i], faulty=faulty, fault_mode=FAULT_MODE))

    # global model
    global_model = Net().to(DEVICE)
    global_params = get_state_ndarrays(global_model)

    # store accuracy
    accuracies = []

    # visualize initial conv filters (first conv layer)
    initial_params = get_state_ndarrays(global_model)
    conv1_init = initial_params[0]  # first parameter is conv1.weight (shape: out_chan, in_chan, k, k)

    for rnd in range(1, NUM_ROUNDS + 1):
        # each client trains and returns weights
        weights_list = []
        sizes = []
        for c in clients:
            w, size = c.fit(global_params)
            weights_list.append(w)
            sizes.append(size)

        # aggregate
        aggregated = aggregate(weights_list, method=AGGREGATOR, trim_ratio=TRIM_RATIO)
        global_params = aggregated  # new global params

        # evaluate global model
        acc = evaluate_global(global_model, global_params)
        accuracies.append(acc)

        if rnd % PRINT_EVERY == 0:
            print(f"Round {rnd:2d} | Aggregator={AGGREGATOR:7s} | Faulty_clients={num_faulty} | Test Acc = {acc*100:5.2f}%")

    # visualize final conv filters
    final_params = global_params
    conv1_final = final_params[0]

    # plot accuracy curve
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, NUM_ROUNDS+1), [a*100 for a in accuracies], marker='o')
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Aggregator={AGGREGATOR} | Faulty fraction={FAULTY_FRACTION} | Fault mode={FAULT_MODE}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy_curve.png", dpi=150)
    print("Saved results/accuracy_curve.png")

    # visualize a few conv filters before/after
    def show_filters(w, name):
        # w shape: (out_channels, in_channels, k, k)
        out_ch = w.shape[0]
        n = min(8, out_ch)
        fig, axes = plt.subplots(2, n//2, figsize=(n, 2))
        axes = axes.flatten()
        for i in range(n):
            filt = w[i,0,:,:]  # display first input channel
            axes[i].imshow(filt, cmap="gray")
            axes[i].axis("off")
        plt.suptitle(name)
        plt.tight_layout()
        return fig

    fig1 = show_filters(conv1_init, "Initial conv1 filters")
    fig1.savefig("results/conv1_initial.png", dpi=150)
    fig2 = show_filters(conv1_final, "Final conv1 filters")
    fig2.savefig("results/conv1_final.png", dpi=150)
    print("Saved results/conv1_initial.png and results/conv1_final.png")

    print("Final test accuracy per round:", ["{:.2f}%".format(a*100) for a in accuracies])

if __name__ == "__main__":
    print(f"Device: {DEVICE}, Aggregator: {AGGREGATOR}, Fault mode: {FAULT_MODE}")
    run_simulation()
