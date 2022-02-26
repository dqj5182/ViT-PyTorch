import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warmup_scheduler

from model.vit.vit import ViT
from dataset.load_cifar import load_cifar
from utils.utils import get_criterion
from utils.dataaug import CutMix, MixUp

# Parser
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
parser.add_argument("--max-epochs", default=1500, type=int)
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

# Device: CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Seed and GPU
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

# Will be used in validation loop (save best model that gives lowest validation loss)
min_valid_loss = np.inf

# For VIT (Error)
if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")


# Load CIFAR-10 data
trainset, testset, trainloader, testloader = load_cifar(args)


# Vision Transformer model
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


#Criterion
criterion = get_criterion(args)


# Optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)


# Cutmix and Mixup (optional)
if args.cutmix:
    cutmix = CutMix(32, beta=1.)
if args.mixup:
    mixup = MixUp(alpha=1.)


# Training and Validation loop
for epoch in range(args.max_epochs):  # loop over the dataset multiple times
    # Training Loop
    train_total = 0
    train_correct = 0
    training_loss = 0.0
    net.train()
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if args.cutmix or args.mixup:
            if args.cutmix:
                img, label, rand_label, lambda_= cutmix((inputs, labels))
            elif args.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = mixup((inputs, labels))
                else:
                    img, label, rand_label, lambda_ = inputs, labels, torch.zeros_like(labels), 1.
            outputs = net(inputs)
            loss = criterion(outputs, label)*lambda_ + criterion(outputs, rand_label)*(1.-lambda_)
        else:
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        training_loss += loss.item()

        # Training Accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    training_acc = 100 * train_correct / train_total
    
    # Validation Loop
    val_loss = 0.0
    val_total = 0
    val_correct = 0
    net.eval()
    with torch.no_grad():
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

    # Capture the best model through training
    if min_valid_loss > (val_loss / len(testloader)):
        print(f'Validation Loss Decreased ({min_valid_loss:.3f}--->{val_loss / len(testloader):.3f}) Saving The Model')
        min_valid_loss = val_loss / len(testloader)
        # Saving State Dict
        torch.save(net.state_dict(), 'vit_saved_model.pth')

print('Finished Training')