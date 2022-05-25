import torch.nn.functional as F

def get_strides(dataset: str):
    if dataset in ['cifar10', 'cifar100']:
        strides = [1, 2, 2, 2]
    elif dataset == 'tiny_imagenet':
        strides = [2, 2, 2, 2]
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return strides

def get_conv1_params(dataset: str):
    if dataset in ['cifar10', 'cifar100']:
        conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    elif dataset == 'tiny_imagenet':
        conv1 = {'kernel_size': 7, 'stride': 1, 'padding': 3}
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return conv1

def get_densenet_conv1_params(dataset: str):
    if dataset in ['cifar10', 'cifar100']:
        conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    elif dataset == 'tiny_imagenet':
        conv1 = {'kernel_size': 7, 'stride': 2, 'padding': 3}
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return conv1

def get_model(moder_str, dataset):
    from . import ResNet18, ResNet34, ResNet50, ResNet101, AlexNetCIFAR, AlexNetTinyImageNet, DenseNet

    if moder_str == 'resnet18':
        return ResNet18
    elif moder_str == 'resnet34':
        return ResNet34
    elif moder_str == 'resnet50':
        return ResNet50
    elif moder_str == 'resnet101':
        return ResNet101
    elif moder_str == 'alexnet':
        if 'cifar' in dataset:
            return AlexNetCIFAR
        else:
            return AlexNetTinyImageNet
    elif moder_str == 'densenet':
        return DenseNet
    else:
        raise AssertionError("network {} is unknown".format(moder_str))

def get_activation(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'softplus':
        return F.softplus
    elif activation == 'swish':
        return F.silu
    else:
        raise AssertionError('activation function {} was not expected'.format(activation))
