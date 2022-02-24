import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import warmup_scheduler

from model.vit.vit import ViT
from model.resnet.resnet import ResNet50
from dataset.load_cifar import load_cifar
from utils.utils import get_model, get_dataset, get_experiment_name, get_criterion
from utils.dataaug import CutMix, MixUp


# Device: CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = ViT(3, 
          10, 
          32, 
          8, 
          0.0, 
          7,
          384,
          384,
          12,
          True
          ).to(device)


n_epochs = 30


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-5)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=base_scheduler)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')























"""
# Device: CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


#net = nn.DataParallel(ResNet50().to(device))

net = ViT(3, 
          10, 
          32, 
          8, 
          0.0, 
          7,
          384,
          384,
          12,
          True
          ).to(device)


n_epochs = 30

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-5)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=base_scheduler)

for epoch in range(n_epochs):  # loop over the dataset multiple times
    training_loss = 0.0
    train_total = 0
    train_correct = 0
    net.train()
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Training Loss
        training_loss += loss.item()

        # Training Accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    training_acc = 100 * train_correct / train_total
    print(training_loss)
"""
"""
val_loss = 0.0
val_total = 0
val_correct = 0
net.eval()
for data in testloader:
    inputs, labels = data[0].to(device), data[1].to(device)

    outputs = net(inputs)
    loss = criterion(outputs, labels)

    val_loss += loss.item()

    # Validation Accuracy
    _, predicted = torch.max(outputs.data, 1)
    val_total += labels.size(0)
    val_correct += (predicted == labels).sum().item()

val_acc = 100 * val_correct / val_total

print(f'Epoch: {epoch + 1} | Training Accuracy: {training_acc:.2f} | Training Loss: {training_loss / len(trainloader):.3f} | Validation Accuracy: {val_acc:.2f} | Validation Loss: {val_loss / len(testloader):.3f}')
"""

print('Finished Training')
