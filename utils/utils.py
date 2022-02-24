import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils.autoaugment import CIFAR10Policy
from utils.dataaug import RandomCropPaste

from model.HCGNet.hcgnet import HCGNet_A1, HCGNet_A2, HCGNet_A3
from model.densenet.densenet import densenet_cifar
from model.dla.dla import DLA
from model.dpn.dpn import DPN26, DPN92
from model.efficientnet.efficientnet import EfficientNetB0
from model.efficientnetv2.efficientnetv2 import effnetv2_s, effnetv2_m, effnetv2_l, effnetv2_xl
from model.mobilenetv3.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.pyramidnet.pyramidnet import pyramidnet164, pyramidnet272
from model.resnet.resnet import ResNet50
from model.resnext.resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from model.vgg.vgg import VGG
from model.vit.vit import ViT


def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")
    return criterion


# This needs severe change
def get_model(args):
    if args.model_name == 'vit':
        net = ViT(
            in_c = args.in_c, 
            num_classes = args.num_classes, 
            img_size=args.size, 
            num_patch_1d=args.patch, 
            dropout=args.dropout, 
            mlp_hidden_dim=args.mlp_hidden,
            num_enc_layers=args.num_layers,
            hidden_dim=args.hidden,
            num_head=args.head,
            is_cls_token=args.is_cls_token
            )
    elif args.model_name == 'densenet':
        net = densenet_cifar()
    elif args.model_name == 'dla':
        net = DLA()
    elif args.model_name == 'dpn':
        net = DPN26()
    elif args.model_name == 'efficientnet':
        net = EfficientNetB0()
    elif args.model_name == 'efficientnetv2':
        net = effnetv2_s()
    elif args.model_name == 'hcgnet':
        net = HCGNet_A1()
    elif args.model_name == 'mobilenetv3':
        net = MobileNetV3_Small()
    elif args.model_name == 'pyramidnet':
        net = pyramidnet164()
    elif args.model_name == 'resnet':
        net = ResNet50()    
    elif args.model_name == 'resnext':
        net = ResNeXt29_2x64d()
    elif args.model_name == 'vgg':
        net = VGG('VGG19')
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")
    print("get model() called", args.model_name)
    return net


def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=32, padding=args.padding)
    ]
    train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10':
            train_transform.append(CIFAR10Policy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10": # CIFAR-10
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=False)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=False)
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds


def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))    
