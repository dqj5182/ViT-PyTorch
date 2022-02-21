import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from model.vit.vit import ViT
from dataset.load_cifar import load_cifar
from utils.utils import get_model, get_dataset, get_experiment_name, get_criterion
from utils.dataaug import CutMix, MixUp

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=350, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=8, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
args = parser.parse_args()


# Seed and GPU
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32


# For VIT (Error)
if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")


# Device: CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
        

# Load CIFAR-10 dataset
# Dataset and Dataloader
trainset, testset = get_dataset(args)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

# Model
net = nn.DataParallel(ViT(in_c = 3, 
                        num_classes = 10, 
                        img_size=32, 
                        num_patch_1d=8, 
                        dropout=0., 
                        mlp_hidden_dim=384*4,
                        num_enc_layers=7,
                        hidden_dim=384,
                        num_head=8,
                        is_cls_token=True
                        ).to(device))
          

# Criterion & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    print("Epoch:", epoch)
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

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


# Test the model
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')