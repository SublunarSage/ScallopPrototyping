import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTParityDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )

  def __len__(self):
    return len(self.mnist_dataset)

  def __getitem__(self, idx):
    # Get data point
    (img, digit) = self.mnist_dataset[idx]
    
    # The target is 1 if even, 0 if odd
    is_even = 1 if digit % 2 == 0 else 0
    
    return (img, is_even)

  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)


def mnist_parity_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTParityDataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTParityDataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTParityDataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTParityDataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTParityNet(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTParityNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit", int, input_mapping=list(range(10)))
    
    # Add rule to detect even numbers (output 1 for even, 0 for odd)
    self.scl_ctx.add_rule("even(1) :- digit(d), d % 2 == 0")
    self.scl_ctx.add_rule("even(0) :- digit(d), d % 2 == 1")

    # The `even` logical reasoning module
    self.parity_detector = self.scl_ctx.forward_function("even", output_mapping=list(range(2)))

  def forward(self, x: torch.Tensor):
    # First recognize the digit
    digit_distr = self.mnist_net(x)  # Tensor batch_size x 10
    
    # Then execute the reasoning module; the result is a size 2 tensor
    # 1st element is probability of odd, 2nd element is probability of even
    return self.parity_detector(digit=digit_distr)  # Tensor batch_size x 2


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTParityNet(provenance, k)
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
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_parity")
  parser.add_argument("--n-epochs", type=int, default=1)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = mnist_parity_loader(data_dir, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)
