# Introduction

PyTorch implementation of efficient image super-resolution models.  

<img src="https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/demo.png" width="100%" height="100%" />

# 

# Requirements

torch == 1.8.1          
torchmetrics  
loguru  
tqdm  

# Supported models

- [CARN](models/carn.py) [^carn]  
- [DRCN](models/drcn.py) [^drcn]  
- [DRRN](models/drrn.py) [^drrn]  
- [EDSR](models/edsr.py) [^edsr]  
- [ESPCN](models/espcn.py) [^espcn]  
- [FSRCNN](models/fsrcnn.py) [^fsrcnn]  
- [IDN](models/idn.py) [^idn]  
- [LapSRN](models/lapsrn.py) [^lapsrn]  
- [SRCNN](models/srcnn.py) [^srcnn]  
- [SRDenseNet](models/srdensenet.py) [^srdensenet]  
- [VDSR](models/vdsr.py) [^vdsr]  

[^carn]: [Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network](https://arxiv.org/abs/1803.08664)  
[^drcn]: [Deeply-Recursive Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1511.04491)  
[^drrn]: [ Image Super-Resolution via Deep Recursive Residual Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Tai_Image_Super-Resolution_via_CVPR_2017_paper.html)  
[^edsr]: [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)  
[^espcn]: [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)  
[^fsrcnn]: [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367)  
[^idn]: [Fast and Accurate Single Image Super-Resolution via Information Distillation Network](https://arxiv.org/abs/1803.09454)  
[^lapsrn]: [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/abs/1704.03915)  
[^srcnn]: [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)  
[^srdensenet]: [Image Super-Resolution Using Dense Skip Connections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)  
[^vdsr]: [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587)  

# How to use

## DDP training (recommend)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## DP training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

# Performances and checkpoints

| Model                                                                                                                    | Year | Train on<sup>1</sup> | Set5            |               | Set14       |               | BSD100      |               |
|:------------------------------------------------------------------------------------------------------------------------:|:----:|:--------------------:|:---------------:|:-------------:|:-----------:|:-------------:|:-----------:|:-------------:|
| 2x                                                                                                                       |      |                      | PSNR (paper/my) | SSIM          | PSNR        | SSIM          | PSNR        | SSIM          |
| [CARN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/carn_2x.pth)             | 2018 | T+B+D                | 37.76/37.90     | 0.9590/0.9605 | 33.52/33.14 | 0.9166/0.9152 | 32.09/32.06 | 0.8978/0.8985 |
| [DRCN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/drcn_2x.pth)             | 2015 | T                    | 37.63/37.85     | 0.9588/0.9604 | 33.04/33.22 | 0.9118/0.916  | 31.85/32.05 | 0.8942/0.8982 |
| [DRRN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/drrn_2x.pth)             | 2017 | T+B                  | 37.74/37.76     | 0.9591/0.9599 | 33.23/33.14 | 0.9136/0.9149 | 32.05/31.99 | 0.8973/0.8974 |
| [EDSR](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/edsr_2x.pth)             | 2017 | D                    | 37.99/37.90     | 0.9604/0.9606 | 33.57/33.22 | 0.9175/0.9163 | 32.16/32.10 | 0.8994/0.899  |
| [ESPCN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/espcn_2x.pth)           | 2016 | I+T                  | n.a./36.85      | n.a./0.9559   | n.a./32.31  | n.a./0.9087   | n.a./31.40  | n.a./0.8897   |
| [FSRCNN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/fsrcnn_2x.pth)         | 2016 | T+G                  | 37.00/37.27     | 0.9558/0.958  | 32.63/32.65 | 0.9088/0.9115 | 31.53/31.67 | 0.8920/0.8934 |
| [IDN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/idn_2x.pth)               | 2018 | T+B                  | 37.83/37.84     | 0.96/0.9604   | 33.30/33.12 | 0.9148/0.9155 | 32.08/32.06 | 0.8985/0.8985 |
| [LapSRN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/lapsrn_2x.pth)         | 2017 | T+B                  | 37.52/37.59     | 0.9591/0.9592 | 32.99/32.96 | 0.9124/0.9138 | 31.80/31.89 | 0.8952/0.8961 |
| [SRCNN](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/srcnn_2x.pth)           | 2014 | I+T                  | 36.66/36.88     | 0.9542/0.9561 | 32.45/32.42 | 0.9067/0.9092 | 31.36/31.50 | 0.8879/0.8907 |
| [SRDenseNet](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/srdensenet_2x.pth) | 2017 | I                    | n.a./37.67      | n.a./0.9596   | n.a./33.05  | n.a./0.9142   | n.a./31.93  | n.a./0.8967   |
| [VDSR](https://github.com/zh320/efficient-image-super-resolution-pytorch/releases/download/v1.0/vdsr_2x.pth)             | 2015 | T+B                  | 37.53/37.74     | 0.9587/0.9598 | 33.03/33.06 | 0.9124/0.9145 | 31.90/31.97 | 0.8960/0.8973 |

[<sup>1</sup> Original training dataset, which are short for B (BSD200), D (DIV2K), G (General100), I (ImageNet), T (T91). In my experiments, the training dataset is T + G + B.]

# Prepare the dataset

```
/train
    /T91
    /General100
    /BSD200
/val
    /Set5
    /Set14
    /BSD100
```

# References