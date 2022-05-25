'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import pickle
from typing import Tuple
import logging
from functools import wraps
import matplotlib.pyplot as plt
from numba import njit, jit
from typing import Dict, List, Tuple
import logging
from collections import OrderedDict
import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

def pytorch_evaluate(net: nn.Module, x: np.ndarray, fetch_keys: List, batch_size: int, x_shape: Tuple = None,
                     output_shapes: Dict = None, to_tensor: bool=False) -> Tuple:

    if output_shapes is not None:
        for key in fetch_keys:
            assert key in output_shapes

    fetches_dict = {}
    fetches = []
    for key in fetch_keys:
        fetches_dict[key] = []

    net.eval()
    num_batch = int(np.ceil(x.shape[0]) / batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for m in range(num_batch):
        # Batch indexes
        begin, end = (m * batch_size, min((m + 1) * batch_size, x.shape[0]))
        input = x[begin:end]
        if x_shape is not None:
            input = input.reshape(x_shape)
        with torch.no_grad():
            outputs_dict = net(torch.from_numpy(input).to(device))
        for key in fetch_keys:
            fetches_dict[key].append(outputs_dict[key].data.cpu().detach().numpy())

    # stack variables together
    for key in fetch_keys:
        fetch = np.vstack(fetches_dict[key])
        if output_shapes is not None:
            fetch = fetch.reshape(output_shapes[key])
        if to_tensor:
            fetch = torch.as_tensor(fetch, device=torch.device(device))
        fetches.append(fetch)

    assert fetches[0].shape[0] == x.shape[0]
    return tuple(fetches)


def boolean_string(s):
    # to use --use_bn True or --use_bn False in the shell. See:
    # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def convert_tensor_to_image(x: np.ndarray):
    """
    :param X: np.array of size (Batch, feature_dims, H, W) or (feature_dims, H, W)
    :return: X with (Batch, H, W, feature_dims) or (H, W, feature_dims) between 0:255, uint8
    """
    X = x.copy()
    X *= 255.0
    X = np.round(X)
    X = X.astype(np.uint8)
    if len(x.shape) == 3:
        X = np.transpose(X, [1, 2, 0])
    else:
        X = np.transpose(X, [0, 2, 3, 1])
    return X

def convert_image_to_tensor(x: np.ndarray):
    """
    :param X: np.array of size (Batch, H, W, feature_dims) between 0:255, uint8
    :return: X with (Batch, feature_dims, H, W) float between [0:1]
    """
    assert x.dtype == np.uint8
    X = x.copy()
    X = X.astype(np.float32)
    X /= 255.0
    X = np.transpose(X, [0, 3, 1, 2])
    return X

def set_logger(log_file):
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler(sys.stdout)]
                        )

def get_all_files_recursive(path, suffix=None):
    files = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(path):
        for file in f:
            if suffix is None:
                files.append(os.path.join(r, file))
            elif '.' + suffix in file:
                files.append(os.path.join(r, file))
    return files

def convert_grayscale_to_rgb(x: np.ndarray) -> np.ndarray:
    """
    Converts a 2D image shape=(x, y) to a RGB image (x, y, 3).
    Args:
        x: gray image
    Returns: rgb image
    """
    return np.stack((x, ) * 3, axis=-1)

def inverse_map(x: dict) -> dict:
    """
    :param x: dictionary
    :return: inverse mapping, showing for each val its key
    """
    inv_map = OrderedDict()
    for k, v in x.items():
        inv_map[v] = k
    return inv_map

def get_image_shape(dataset: str) -> Tuple[int, int, int]:
    if dataset in ['cifar10', 'cifar100']:
        return 32, 32, 3
    elif dataset == 'tiny_imagenet':
        return 64, 64, 3
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))

def get_num_classes(dataset: str) -> int:
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'tiny_imagenet':
        return 200
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))

def get_max_train_size(dataset: str) -> int:
    if dataset in ['cifar10', 'cifar100']:
        return 50000
    elif dataset == 'svhn':
        return 72000
    elif dataset == 'tiny_imagenet':
        return 100000
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))

def get_parameter_groups(net: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    no_decay = dict()
    decay = dict()
    for name, m in net.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            decay[name + '.weight'] = m.weight
            decay[name + '.bias'] = m.bias
        elif isinstance(m, nn.BatchNorm2d):
            no_decay[name + '.weight'] = m.weight
            no_decay[name + '.bias'] = m.bias
        else:
            if hasattr(m, 'weight'):
                no_decay[name + '.weight'] = m.weight
            if hasattr(m, 'bias'):
                no_decay[name + '.bias'] = m.weight

    # remove all None values:
    del_items = []
    for d, v in decay.items():
        if v is None:
            del_items.append(d)
    for d in del_items:
        decay.pop(d)

    del_items = []
    for d, v in no_decay.items():
        if v is None:
            del_items.append(d)
    for d in del_items:
        no_decay.pop(d)

    return decay, no_decay

def force_lr(optimizer, lr):
    """ Force a specific learning rate to all of the optimizer's weights"""
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr

def calc_acc_precision_recall(inferred_non_member, inferred_member):
    logger = logging.getLogger()
    member_acc = np.mean(inferred_member == 1)
    non_member_acc = np.mean(inferred_non_member == 0)
    acc = (member_acc * len(inferred_member) + non_member_acc * len(inferred_non_member)) / (len(inferred_member) + len(inferred_non_member))
    precision, recall, f_score, true_sum = precision_recall_fscore_support(
        y_true=np.concatenate((np.zeros(len(inferred_non_member)), np.ones(len(inferred_member)))),
        y_pred=np.concatenate((inferred_non_member, inferred_member)),
    )
    logger.info('member acc: {}, non-member acc: {}, balanced acc: {}, precision/recall(member): {}/{}, precision/recall(non-member): {}/{}'
                .format(member_acc, non_member_acc, acc, precision[1], recall[1], precision[0], recall[0]))

def load_state_dict(model: nn.Module, path: str, device='cpu') -> int:
    global_state = torch.load(path, map_location=torch.device(device))
    if 'best_net' in global_state:
        global_state = global_state['best_net']
    model.load_state_dict(global_state)
    model.to(device)
    return 1

def save_to_path(path: str, x: np.ndarray, overwrite=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or overwrite:
        np.save(path, x)

def normalize(x, rgb_mean, rgb_std):
    """
    :param x: np.ndaaray of image RGB of (3, W, H), normalized between [0,1]
    :param rgb_mean: Tuple of (RED mean, GREEN mean, BLUE mean)
    :param rgb_std: Tuple of (RED std, GREEN std, BLUE std)
    :return np.ndarray transformed by x = (x-mean)/std
    """
    transform = transforms.Normalize(rgb_mean, rgb_std)
    x_tensor = torch.tensor(x)
    x_new = transform(x_tensor)
    x_new = x_new.cpu().numpy()
    return x_new
