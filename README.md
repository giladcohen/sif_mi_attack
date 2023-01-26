# SIF MI Attack
This repo reproduces all the reuslts shown in our paper:

[**Membership Inference Attack Using Self Influence Functions**](https://arxiv.org/abs/2205.13680)

# Init project
Run in the project dir:
```
source ./init_project.sh
```
# Train target models
Train AlexNet/Resnet/DenseNet networks for cifar10, cifar100, and tiny_imagenet using src/train_target_model.py.

Example 1: Train CIFAR-10 on ResNet18 with 25k sample points (M-7 target model in the paper) with augmentations:
```
python src/train_target_model.py --checkpoint_dir /tmp/mi/cifar10/resnet18/s_25k_w_aug --dataset cifar10 --train_size 0.5 --augmentations True --net resnet18
```
Example 2: Differential Privacy training of Tiny ImageNet on ResNet18 with 25k sample points (M-7 target model in the paper):
```
python src/train_target_model_dp.py --checkpoint_dir /tmp/mi/tiny_imagenet/resnet18/s_25k_dp --dataset tiny_imagenet --train_size 0.25 --augmentations False --net resnet18
```

# Run MI attack
Attack a target model using Gap, Black-box, Boundary distance, or our SIF, avgSIF, adaSIF attacks.

To run an attack on a target model, specify its checkpoint dir and the required attack.
    
Examples:

1) Run our vanilla SIF attack:
```
python src/attack.py --checkpoint_dir /tmp/mi/cifar10/resnet18/s_25k_w_aug --attack self_influence --output vanilla_sif
```
2) Run adaSIF with recursive depth (d) of 8 and 8 iterations (d=8 and r=8 in our paper):
```
python src/attack.py --checkpoint_dir /tmp/mi/cifar10/resnet18/s_25k_w_aug --attack self_influence --adaptive True --rec_dep 8 --r 8 --output adaSIF
```    
3) Run avgSIF:
```
python src/attack.py --checkpoint_dir /tmp/mi/cifar10/resnet18/s_25k_w_aug --attack self_influence --average True --output avgSIF
```    
4) Run baselines:
```
python src/attack.py --checkpoint_dir /tmp/mi/cifar10/resnet18/s_25k_w_aug --attack <BASELINE>
```    
where `<BASELINE>` can be one of the following: gap, black_box, boundary_distance.

Notice that in our paper we fit and infer the Boundary distance attack on 500 and 2500 samples, respectively. To do the same, run:
```
python src/attack.py --checkpoint_dir /tmp/mi/cifar10/resnet18/s_25k_w_aug --attack boundary_distance --fast True
```    
# Run MI attack on pretrained models
To compare our results to the white-box attack in [https://arxiv.org/abs/1812.00910](https://arxiv.org/abs/1812.00910), we use the same pretrained models they employed to train CIFAR-100 in their paper (AlexNet, ResNet110, and DenseNet). These pretrained models can be downloded from [GitHub repo](https://github.com/bearpaw/pytorch-classification). Since ResNet110 model weights cannot be loaded successfully, we trained the same architecture from scratch via:
```
cd pytorch_classification
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint /tmp/mi/cifar100/resnet110_ref_paper
```
This script runs our adaSIF attack on the pretrained DenseNet model for CIFAR-100:
```
python src/attack_ref_cifar100.py --checkpoint_dir /tmp/mi/cifar100/densenet_ref_paper --arch densenet --attack self_influence --adaptive True --rec_dep 8 --r 8
```
