import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import save_file

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

# Number of letter classes (A-Z)
NUM_CLASSES = 26

# Transform for EMNIST (which contains alphabetic characters)
emnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])

def emnist_loader(data_dir, batch_size_train, batch_size_test):
    # EMNIST dataset contains handwritten letters
    # We're using the 'letters' split that has 26 classes (A-Z)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(
            data_dir,
            split='letters',  # This specifically selects the letters dataset
            train=True,
            download=True,
            transform=emnist_img_transform,
        ),
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(
            data_dir,
            split='letters',
            train=False,
            download=True,
            transform=emnist_img_transform,
        ),
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


class AlphabetNet(nn.Module):
    def __init__(self):
        super(AlphabetNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        # Change output to 26 for A-Z
        self.fc2 = nn.Linear(1024, NUM_CLASSES)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class AlphabetWithScallop(nn.Module):
    def __init__(self):
        super(AlphabetWithScallop, self).__init__()

        # Alphabet Recognition Network
        self.alphabet_net = AlphabetNet()

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance="difftopkproofs", k=3)
        
        # Define the letter relation instead of digit
        # EMNIST classes are 1-indexed for letters (1=A, 2=B, ..., 26=Z)
        # But our model outputs will be 0-indexed, so we map accordingly
        self.scl_ctx.add_relation("letter", int, input_mapping=[(i,) for i in range(NUM_CLASSES)])
        
        # We can use a simple identity rule similar to the original
        self.scl_ctx.add_rule("result(x) = letter(x)")

        # The logical reasoning module
        self.identity_fn = self.scl_ctx.forward_function("result", [(i,) for i in range(NUM_CLASSES)])

    def forward(self, x: torch.Tensor):
        letter_probs = self.alphabet_net(x)
        return self.identity_fn(letter=letter_probs)


def bce_loss(output, ground_truth):
    # Adjust to work with 26 classes instead of 10
    gt = torch.stack([torch.tensor([1.0 if i == t-1 else 0.0 for i in range(NUM_CLASSES)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    # EMNIST is 1-indexed, so we need to adjust the ground truth
    # Subtract 1 from ground_truth to convert to 0-indexed
    adjusted_gt = ground_truth - 1
    return F.nll_loss(output, adjusted_gt)


class Trainer():
    def __init__(self, train_loader, test_loader, learning_rate, loss):
        self.network = AlphabetWithScallop()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        if loss == "nll":
            self.loss = nll_loss
        elif loss == "bce":
            self.loss = bce_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for (data, target) in iter:
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")

    def test(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for (data, target) in iter:
                output = self.network(data)
                test_loss += self.loss(output, target).item()
                
                # EMNIST classes are 1-indexed, so adjust predictions accordingly
                pred = output.data.max(1, keepdim=True)[1] + 1
                correct += pred.eq(target.data.view_as(pred)).sum()
                
                perc = 100. * correct / num_items
                iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

    def save_model(self, save_path):
        """Save the model weights using safetensors format"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract state dict from the model
        state_dict = self.network.state_dict()
        
        # Save using safetensors
        save_file(state_dict, save_path)
        print(f"Model saved to {save_path}")
        
    def train(self, n_epochs, save_path=None):
        self.test(0)
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.test(epoch)
            
        # Save the model after training if a save path is provided
        if save_path:
            self.save_model(save_path)


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("emnist_letters")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--loss-fn", type=str, default="bce")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-model", type=str, default="model/alphabet_model.safetensors",
                        help="Path to save the trained model in safetensors format")
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    loss_fn = args.loss_fn
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

    # Dataloaders
    train_loader, test_loader = emnist_loader(data_dir, batch_size_train, batch_size_test)

    # Create trainer and train
    trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn)
    trainer.train(n_epochs, args.save_model)