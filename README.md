# Introduction
PyTorch implementation of efficient image super-resolution models.  

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
Benchmarks are coming.

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