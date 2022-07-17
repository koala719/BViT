# BViT: Broad Attention based Vision Transformer
This repository contains the official implementation of [Broad Attention based Vision Transformer](https://arxiv.org/abs/2202.06268) 
![alt](https://github.com/koala719/BViT/blob/main/figs/overall_b.png)
Recent works have demonstrated that transformer can achieve promising performance in computer vision, by exploiting the relationship among image patches with self-attention. While they only consider the attention in a single feature layer, but ignore the complementarity of attention in different layers. In this paper, we propose the broad attention to improve the performance by incorporating the attention relationship of different layers for vision transformer, which is called BViT. The broad attention is implemented by broad connection and parameter-free attention. Broad connection of each transformer layer promotes the transmission and integration of information for BViT. Without introducing additional trainable parameters, parameter-free attention jointly focuses on the already available attention information in different layers for extracting useful information and building their relationship. Experiments on image classification tasks demonstrate that BViT delivers state-of-the-art accuracy of 74.8%/81.6% top-1 accuracy on ImageNet with 5M/22M parameters. Moreover, we transfer BViT to downstream object recognition benchmarks to achieve
98.9% and 89.9% on CIFAR10 and CIFAR100 respectively that exceed ViT with fewer parameters. For the generalization test, the broad attention in Swin Transformer and T2T-ViT also bring an improvement of more than 1%. To sum up, broad attention is promising to promote the performance of attention-based models.

## Model Results
|  Model   | Top1-Acc(%)  |  Params (M)   | Flops(G)  | Model  |
|  ----  | ----  |  ----  | ----  | ----  |
| BViT-5M  | 74.8 |  5.7  | 1.2  | [Download](https://pan.baidu.com/s/1q02tHE9Jk3M9PcIiK4vrdg?pwd=65q1)   |
| BViT-22M  | 81.6 |  22.1  | 4.7  | [Download](https://pan.baidu.com/s/1G_Zh-qDbAtcYvVYLXoF-Nw?pwd=dbgi)   |

## Evaluation
Install PyTorch 1.7.0+ and torchvision 0.8.1+ and timm 0.3.2.

### Training
To evaluate a pre-trained BViT on ImageNet val with a single GPU run:
```
python train_hp.py --eval --resume /path/to/weight --data-path /path/to/imagenet
```

### Training
To train BViT on ImageNet with 4 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_hp.py --arch BViT-5 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```



## Bibtex
If you find BViT is helpful in your project, please consider citing our paper.
```
@article{li2022bvit,
  title={BViT: Broad Attention based Vision Transformer},
  author={Li, Nannan and Chen, Yaran and Li, Weifan and Ding, Zixiang and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2202.06268},
  year={2022}
}
```

## Acknowledgements
The codes are inspired by [timm](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [ViT-torch](https://github.com/lucidrains/vit-pytorch).

