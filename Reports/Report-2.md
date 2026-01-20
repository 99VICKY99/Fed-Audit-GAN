B.Tech. Major Project (CS499) – Weekly Progress Report 2
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [23 August 2025 – 28 August 2025]
1. Progress Summary*
Upon further studying of the paper, our guide suggested to implement the different algorithms of Federated Learning and then implement the code for the frameworks suggested as per the paper. Hence, we started with the easiest algorithm, FedAvg. We found a repository on Github.com (https://github.com/naderAsadi/FedAvg) where the author implemented FedAvg on MNIST dataset in both IID setting and non-IID settings (with shards). To further our goals, we modified the dataset to be CIFAR10, and also included the sampler that implements Dirichlet. After this, we ran some tests on FedAvg with different scenarios such as α = 0.1 and α = 0.3 etc. It was evident that Federated Learning in non-IID setting leads to poor accuracy as running the algorithm in 20 epochs in IID setting would lead to accuracy around 97%. Meanwhile, the non-IID settings (with both shards and Dirichlet based distribution) would lead to very low accuracy around 31-38% when the dataset is Cifar10, which does not even surpass 40% of accuracy. This has given us a better understanding of the underlying process of FL and with this, we may prepare the new frameworks built upon the older ones.
2. Tasks Completed This Week*
This was the Project structure we were working upon:
FedAvg/
├── data/               # Dataset storage and partitioning
├── models/             # Model definitions
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
├── environment.yml     # Conda environment configuration
├── fed_avg.py          # Main script for training and evaluation
├── sweep.yaml          # Hyperparameter sweep configuration for WandB
└── utils.py            # Utility functions



Now, here were the changes that we had suggested to test it on other datasets such as Cifar10 and include Dirichlet based sampling:

data/cifar.py

# FedAvg/data/cifar.py
from typing import Optional
import numpy as np
import torch
from torchvision import datasets, transforms

class CIFAR10Dataset(datasets.CIFAR10):

N_CLASSES = 10

def __init__(self, root: str, train: bool):
transform = transforms.Compose(
[
transforms.ToTensor(),
transforms.Normalize(
(0.4914, 0.4822, 0.4465), # CIFAR-10 mean
(0.2023, 0.1994, 0.2010), # CIFAR-10 std
),
]
)
super().__init__(root=root, train=train, download=True, transform=transform)

def __getitem__(self, index):
# `self.data` is a numpy array and `self.targets` could be list or tensor
x, y = self.data[index], self.targets[index]
x = self.transform(x)
return x, y

Enter a function for Dirichlet based sampling

def _sample_dirichlet(self, alpha: float) -> Dict[int, List[int]]:
"""
Dirichlet-based partition. For each class k, split its indices among clients using a Dirichlet distribution with parameter alpha.
Lower alpha -> more skewed (heterogeneous).
"""
labels = np.array(self.dataset.targets)
n_classes = len(np.unique(labels))
dict_users = {i: [] for i in range(self.n_clients)}

# For each class, split indices
for k in range(n_classes):
idx_k = np.where(labels == k)[0]
if len(idx_k) == 0:
continue
np.random.shuffle(idx_k)
# sample proportions for n_clients
proportions = np.random.dirichlet(alpha * np.ones(self.n_clients))
# split according to proportions
# compute split points
counts = (proportions * len(idx_k)).astype(int)

# To ensure all elements are assigned (fix rounding)
# distribute leftover indices (due to flooring) to clients with largest fractional parts
remainder = len(idx_k) - counts.sum()
if remainder > 0:
# fractional parts
fractional = proportions * len(idx_k) - counts
add_idx = np.argsort(-fractional)[:remainder]
for j in add_idx:
counts[j] += 1

start = 0
for client_id, c in enumerate(counts):
if c > 0:
dict_users[client_id].extend(idx_k[start : start + c].tolist())
start += c

# convert to numpy arrays
for i in range(self.n_clients):
dict_users[i] = np.array(dict_users[i], dtype=int)

return dict_users

Update the Model to adapt to the Cifar10 dataset

class CNN(nn.Module):
def __init__(self, n_channels=1, n_classes=10):
"""
CNN that works for both MNIST (1 channel, 28x28) and CIFAR-10 (3 channels, 32x32)
"""
super(CNN, self).__init__()

# Convolutional layers
self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
self.pool = nn.MaxPool2d(2, 2)

# Fully connected layers (depends on input image size!)
if n_channels == 1: # MNIST (28x28 → 7x7 after two poolings)
fc_input_dim = 64 * 7 * 7
else: # CIFAR-10 (32x32 → 8x8 after two poolings)
fc_input_dim = 64 * 8 * 8

self.fc1 = nn.Linear(fc_input_dim, 128)
self.fc2 = nn.Linear(128, n_classes)

def forward(self, x):
x = self.pool(F.relu(self.conv1(x))) # (B, 32, H/2, W/2)
x = self.pool(F.relu(self.conv2(x))) # (B, 64, H/4, W/4)
x = x.view(x.size(0), -1) # flatten
x = F.relu(self.fc1(x))
x = self.fc2(x)
return F.log_softmax(x, dim=1)


Update the Argument Parser to recognize which dataset to use

def arg_parser() -> argparse.Namespace:
parser = argparse.ArgumentParser()

parser.add_argument("--data_root", type=str, default="../datasets/")
parser.add_argument("--model_name", type=str, default="cnn")

# non_iid modes: 0 = IID, 1 = Shards, 2 = Dirichlet
parser.add_argument("--non_iid", type=int, default=1) # 0: IID, 1: Shards, 2: Dirichlet
parser.add_argument("--alpha", type=float, default=None) # Dirichlet concentration parameter (if using dirichlet)
parser.add_argument("--n_clients", type=int, default=100)
parser.add_argument("--n_shards", type=int, default=200)
parser.add_argument("--frac", type=float, default=0.1)

parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--n_client_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optim", type=str, default="sgd")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--log_every", type=int, default=1)
parser.add_argument("--early_stopping", type=int, default=1)

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--wandb_project", type=str, default="FedAvg")
parser.add_argument("--exp_name", type=str, default="exp")

return parser.parse_args()

3. Plan for the Upcoming Week*
Now that we have the codebase with ourselves, we can extend it to other algorithms just like we did in case of using CIFAR10 instead of MNIST and including a dirichlet based distribution of dataset among the clients. Hence, we shall look into the underlying aspects of the paper implementation and then translate them into the code afterwards.
4. Additional Notes*

Though the codebase was helpful, it still has a lot of questions that need to be answered. We need to understand CNNs and dataset distribution in greater depth as we did not know what was the meaning of Dirichlet and Shards before. With that, we would need to know how this entire process works and then also understand where each module plays its role.





Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)



Supervisor’s Signature: ___________________
