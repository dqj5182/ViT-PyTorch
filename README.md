# VIT_CNN_CIFAR_10_FROM_SCRATCH

Folder Structure
```bash
.
├── model
│   ├── HCGNet
│   ├── densenet
│   ├── dla
│   ├── dpn
│   ├── efficientnet
│   ├── efficientnetV2
│   ├── mobilenetV3
│   ├── pyramidnet
│   ├── resnet
│   ├── resnext
│   ├── vgg
│   └── vit
├── notebooks
│   └── Pretrained_Vision_Transformer_w_o_PyTorch_Lightning.ipynb
├── utils
│   ├── autoaugment.py
│   ├── dataaug.py
│   └── utils.py
├── main.py
└── vit_saved_model.pth          
```

</br>

[General Option] Train the any model with CIFAR-10 dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --label-smoothing --autoaugment --model-name [name of the model]
```
</br>

[Option VIT] Train VIT model with CIFAR-10 dataset (default):
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --label-smoothing --autoaugment
```
</br>

[Option ResNet] Train ResNet model with CIFAR-10 dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --label-smoothing --autoaugment --model-name resnet
```
</br>
CUDA_VISIBLE_DEVICES=0 means that we are assigning GPU0 to be our CUDA device
</br>
</br>

## Pretrained Models
Epoch: 200 (CNN based), Epoch: 1500 (VIT)
Model | Pretrained (.pth) | Notebook | Paper | Accuracy
--- | --- | --- | --- | ---
Vision Transformer | [Google Drive](https://drive.google.com/file/d/1EsMUTy-IwISqsbFDyn_6pG4XExA6mE2P/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/1vPFNCh2xMp7fgckaY9fGt0zkdPCq3rLO?usp=sharing) | [Arxiv](https://arxiv.org/pdf/2010.11929.pdf) | 90.61%
ResNet-50 | [Google Drive](https://drive.google.com/file/d/1FNsXhHEpEKiZQFgk8vBppmGIABEYXb88/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/1NSBEJSnQ4wgt6_bvPPKlbyxASZcZgfH6?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1512.03385.pdf) | 95.36%
ResNeXT | [Google Drive](https://drive.google.com/file/d/1LBn_AVSaQv1O0LaGN8JUWwwyFQ5_mIwl/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/1yTg8EmAi2yVpKBthnh-S47sG6dPEoQxD?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1611.05431.pdf) | 94.91%
HCGNet | [Google Drive](https://drive.google.com/file/d/1jNDtVZTB9DAWB9ZzaVUbF-sE2-LsMq1O/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/10Ey9Dc2Va3b2O-L15a27AqOXj7cYtMtm?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1908.09699.pdf) | 94.76%
DenseNet | [Google Drive](https://drive.google.com/file/d/1NVe2wwJLxL1XH1tbenrVU7iV5Tx3DuyL/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/197uig6UEecpbLIswSKps4Fu4oaRN1yuS?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1608.06993.pdf) | 94.59%
VGG-19 | [Google Drive](https://drive.google.com/file/d/1kNcRFOpmotVaKER9ThTpYIdWQg_p8_Ab/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/1bxbpmPQsnb1DzwQGz1m9QTacUDwBkGh7?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1409.1556.pdf) | 94.37%
VGG-11 | [Google Drive](https://drive.google.com/file/d/1h4C4WQHqVhGOHKyCkk41unA_A9Ao4R3h/view?usp=sharing) | [Google Colab](https://colab.research.google.com/drive/1DOPFU3J6_mxVDCvJLH9lBL1rbH84LGrD?usp=sharing) | [Arxiv](https://arxiv.org/pdf/1409.1556.pdf) | 92.72%
</br>

## From Scratch VS PyTorch Lightning
Please visit [my another GitHub repo](https://github.com/dqj5182/VIT_CNN_CIFAR_10_PyTorch_Lightning) for PyTorch Lightning version of this code. They work pretty much same but in different format.

## Reference
Many of the codes for VIT models, training, and testing are from [ViT-CIFAR repo](https://github.com/omihub777/ViT-CIFAR) by [omihub777](https://github.com/omihub777)</br>
Some of the codes for VIT models are from [vision_transformer repo](https://github.com/google-research/vision_transformer) by [google-research](https://github.com/google-research)</br>
EfficientNet from [EfficientNet-PyTorch repo](https://github.com/lukemelas/EfficientNet-PyTorch) by [lukemelas](https://github.com/lukemelas)</br>
MobileNetV3 from [mobilenetv3 repo](https://github.com/xiaolai-sqlai/mobilenetv3) by [xiaolai-sqlai](https://github.com/xiaolai-sqlai)</br>
Most of the other models from [pytorch-cifar repo](https://github.com/kuangliu/pytorch-cifar) by [kuangliu](https://github.com/kuangliu/pytorch-cifar)
