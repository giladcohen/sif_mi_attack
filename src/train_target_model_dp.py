"""Train Target models with differential privacy
Datasets: CIFAR-10, CIFAR-100, Tiny ImageNet
Architectures: AlexNet, Resnet18, DenseNet
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torchvision import transforms
from typing import Tuple, Any, Dict
import copy
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import time
import sys
import logging

from datasets.train_val_test_data_loaders import get_test_loader, get_train_valid_loader
from utils import boolean_string, get_image_shape, set_logger
from models.utils import get_strides, get_conv1_params, get_densenet_conv1_params, get_model

parser = argparse.ArgumentParser(description='Training DP networks using PyTorch')
parser.add_argument('--checkpoint_dir', default='/tmp/mi/tiny_imagenet/resnet18/s_25k_dp', type=str, help='checkpoint dir')

# dataset
parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='dataset: cifar10, cifar100, svhn, tiny_imagenet')
parser.add_argument('--train_size', default=0.25, type=float, help='Fraction of train size out of entire trainset')
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size out of entire trainset')
parser.add_argument('--augmentations', default=False, type=boolean_string, help='whether to include data augmentations')

# architecture:
parser.add_argument('--net', default='resnet18', type=str, help='network architecture')
parser.add_argument('--activation', default='relu', type=str, help='network activation: relu, softplus, or swish')

# optimization:
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--num_workers', default=4, type=int, help='Data loading threads')
parser.add_argument('--metric', default='accuracy', type=str, help='metric to optimize. accuracy or sparsity')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--max_physical_batch_size', default=64, type=int, help='batch size')

# differential privacy
parser.add_argument('--epsilon', default=50.0, type=float, help='Noise level. Smaller -> more noise')
parser.add_argument('--max_grad_norm', default=1.2, type=float, help='Max gradient norm')

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

# dumping args to txt file
os.makedirs(args.checkpoint_dir, exist_ok=True)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
log_file = os.path.join(args.checkpoint_dir, 'log.log')
batch_size = args.batch_size

set_logger(log_file)

# importing opacus just now not to override logger
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

logger = logging.getLogger()
if args.metric == 'accuracy':
    WORST_METRIC = 0.0
    metric_mode = 'max'
elif args.metric == 'loss':
    WORST_METRIC = np.inf
    metric_mode = 'min'
else:
    raise AssertionError('illegal argument metric={}'.format(args.metric))

rand_gen = np.random.RandomState(int(time.time()))  # we want different nets for ensemble, for reproducibility one
# might want to replace the time with a contant.
train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

# Data
logger.info('==> Preparing data..')
dataset_args = dict()
if not args.augmentations:
    dataset_args['train_transform'] = transforms.ToTensor()

trainloader, valloader, train_inds, val_inds = get_train_valid_loader(
    dataset=args.dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    rand_gen=rand_gen,
    train_size=args.train_size,
    valid_size=args.val_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)
testloader = get_test_loader(
    dataset=args.dataset,
    dataset_args=dataset_args,
    batch_size=batch_size,
    num_workers=args.num_workers,
    pin_memory=device=='cuda'
)

img_shape = get_image_shape(args.dataset)
classes = trainloader.dataset.classes
num_classes = len(classes)
train_size = len(trainloader.dataset)
val_size   = len(valloader.dataset)
test_size  = len(testloader.dataset)

y_train    = np.asarray(trainloader.dataset.targets)
y_val      = np.asarray(valloader.dataset.targets)
y_test     = np.asarray(testloader.dataset.targets)

# dump to dir:
np.save(os.path.join(args.checkpoint_dir, 'y_train.npy'), y_train)
np.save(os.path.join(args.checkpoint_dir, 'y_val.npy'), y_val)
np.save(os.path.join(args.checkpoint_dir, 'y_test.npy'), y_test)
np.save(os.path.join(args.checkpoint_dir, 'train_inds.npy'), train_inds)
np.save(os.path.join(args.checkpoint_dir, 'val_inds.npy'), val_inds)

# Model
logger.info('==> Building model..')
net_cls = get_model(args.net, args.dataset)
if 'resnet' in args.net:
    conv1 = get_conv1_params(args.dataset)
    strides = get_strides(args.dataset)
    net = net_cls(num_classes=num_classes, activation=args.activation, conv1=conv1, strides=strides)
elif args.net == 'alexnet':
    net = net_cls(num_classes=num_classes, activation=args.activation)
elif args.net == 'densenet':
    assert args.activation == 'relu'
    conv1 = get_densenet_conv1_params(args.dataset)
    net = net_cls(growth_rate=6, num_layers=52, num_classes=num_classes, drop_rate=0.0, conv1=conv1)
else:
    raise AssertionError('Does not support non Resnet architectures')

# errors = ModuleValidator.validate(net, strict=False)
# logger.infofo('error before fixing are:\n{}'.format(errors))
net = ModuleValidator.fix(net)
# errors = ModuleValidator.validate(net, strict=False)
# logger.info('error after fixing are:\n{}'.format(errors))

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
summary(net, (img_shape[2], img_shape[0], img_shape[1]))

optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
ce_criterion = nn.CrossEntropyLoss()

def loss_func(inputs, targets, kwargs=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    losses = {}
    outputs = net(inputs)
    losses['cross_entropy'] = ce_criterion(outputs['logits'], targets)
    losses['loss'] = losses['cross_entropy']
    return outputs, losses

def pred_func(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    _, preds = outputs['logits'].max(1)
    preds = preds.cpu().numpy()
    return preds


delta = 0.5 * (1 / train_size)
privacy_engine = PrivacyEngine()
net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
    module=net,
    optimizer=optimizer,
    data_loader=trainloader,
    epochs=args.epochs,
    target_epsilon=args.epsilon,
    target_delta=delta,
    max_grad_norm=args.max_grad_norm,
)
logger.info(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")


def train():
    """Train and validate"""
    # Training
    global global_step
    global net

    net.train()
    train_loss = 0
    predicted = []
    labels = []

    with BatchMemoryManager(
            data_loader=trainloader,
            max_physical_batch_size=args.max_physical_batch_size,
            optimizer=optimizer
    ) as memory_safe_data_loader:
        for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):  # train a single step
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, loss_dict = loss_func(inputs, targets)
            # print(loss_dict)

            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = pred_func(outputs)
            targets_np = targets.cpu().numpy()
            predicted.extend(preds)
            labels.extend(targets_np)
            num_corrected = np.sum(preds == targets_np)
            acc = num_corrected / targets.size(0)

            if global_step % 10 == 0:  # sampling, once ever 10 train iterations
                for k, v in loss_dict.items():
                    train_writer.add_scalar('losses/' + k, v.item(), global_step)
                train_writer.add_scalar('metrics/acc', 100.0 * acc, global_step)
                train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

    N = batch_idx + 1
    train_loss = train_loss / N
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    logger.info('Epoch #{} (TRAIN): loss={}\tacc={:.2f}'.format(epoch + 1, train_loss, train_acc))

def validate():
    global global_state
    global best_metric
    global net

    net.eval()
    val_loss = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, loss_dict = loss_func(inputs, targets)
            preds = pred_func(outputs)

            loss = loss_dict['loss']
            val_loss += loss.item()
            predicted.extend(preds)

    N = batch_idx + 1
    val_loss    = val_loss / N
    predicted = np.asarray(predicted)
    val_acc = 100.0 * np.mean(predicted == y_val)

    val_writer.add_scalar('losses/loss', val_loss, global_step)
    val_writer.add_scalar('metrics/acc', val_acc, global_step)

    if args.metric == 'accuracy':
        metric = val_acc
    elif args.metric == 'loss':
        metric = val_loss
    else:
        raise AssertionError('Unknown metric for optimization {}'.format(args.metric))

    if (args.metric == 'accuracy' and metric > best_metric) or (args.metric == 'loss' and metric < best_metric):
        best_metric = metric
        logger.info('Found new best model. Saving...')
        save_global_state()
    logger.info('Epoch #{} (VAL): loss={}\tacc={:.2f}\tbest_metric({})={}'.format(epoch + 1, val_loss, val_acc, args.metric, best_metric))

def test():
    global net

    net.eval()
    test_loss = 0
    predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, loss_dict = loss_func(inputs, targets)
            preds = pred_func(outputs)

            loss = loss_dict['loss']
            test_loss += loss.item()
            predicted.extend(preds)

    N = batch_idx + 1
    test_loss = test_loss / N
    predicted = np.asarray(predicted)
    test_acc = 100.0 * np.mean(predicted == y_test)

    test_writer.add_scalar('losses/loss', test_loss, global_step)
    test_writer.add_scalar('metrics/acc', test_acc, global_step)

    logger.info('Epoch #{} (TEST): loss={}\tacc={:.2f}'.format(epoch + 1, test_loss, test_acc))

def save_global_state():
    global global_state
    global_state['best_net'] = copy.deepcopy(net).state_dict()
    global_state['best_metric'] = best_metric
    global_state['epoch'] = epoch
    global_state['global_step'] = global_step
    torch.save(global_state, CHECKPOINT_PATH)

def save_current_state():
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'ckpt_epoch_{}.pth'.format(epoch)))

def flush():
    train_writer.flush()
    val_writer.flush()
    test_writer.flush()
    logger.handlers[0].flush()

def load_best_net():
    global net
    global_state = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
    net.load_state_dict(global_state['best_net'])

if __name__ == "__main__":
    best_metric = WORST_METRIC
    epoch = 0
    global_step = 0
    global_state = {}

    logger.info('Testing epoch #{}'.format(epoch + 1))
    test()

    logger.info('Start training from epoch #{} for {} epochs'.format(epoch + 1, args.epochs))
    for epoch in tqdm(range(epoch, epoch + args.epochs)):
        train()
        validate()
        if epoch % 10 == 0 and epoch > 0:
            test()
        if epoch % 100 == 0 and epoch > 0:  # increase value for large models
            save_current_state()  # once every 10 epochs, save network to a new, distinctive checkpoint file
    save_current_state()

    # getting best metric, loading best net
    load_best_net()
    test()
    flush()
