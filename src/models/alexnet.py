import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

class AlexNetImageNet(nn.Module):
    def __init__(self, num_classes: int, activation: str = 'relu', field: str = None) -> None:
        super().__init__()
        self.activation = get_activation(activation)
        self.field = field

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        net = {}

        # features
        out = self.max_pool(self.activation(self.conv1(x)))
        out = self.max_pool(self.activation(self.conv2(out)))
        out = self.activation(self.conv3(out))
        out = self.activation(self.conv4(out))
        out = self.max_pool(self.activation(self.conv5(out)))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        net['embeddings'] = out

        # classifier
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out = self.linear3(out)
        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        if self.field is None:
            return net
        else:
            return net[self.field]


class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes: int, activation: str = 'relu', field: str = None) -> None:
        super().__init__()
        self.activation = get_activation(activation)
        self.field = field

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(256 * 2 * 2, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        net = {}

        # features
        out = self.max_pool(self.activation(self.conv1(x)))
        out = self.max_pool(self.activation(self.conv2(out)))
        out = self.activation(self.conv3(out))
        out = self.activation(self.conv4(out))
        out = self.max_pool(self.activation(self.conv5(out)))
        out = out.view(out.size(0), 256 * 2 * 2)  # torch.flatten(out, 1)
        net['embeddings'] = out

        # classifier
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out = self.linear3(out)
        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        if self.field is None:
            return net
        else:
            return net[self.field]


class AlexNetTinyImageNet(nn.Module):
    def __init__(self, num_classes: int, activation: str = 'relu', field: str = None) -> None:
        super().__init__()
        self.activation = get_activation(activation)
        self.field = field

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(256 * 2 * 2, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        net = {}

        # features
        out = self.max_pool(self.activation(self.conv1(x)))
        out = self.max_pool(self.activation(self.conv2(out)))
        out = self.activation(self.conv3(out))
        out = self.activation(self.conv4(out))
        out = self.max_pool(self.activation(self.conv5(out)))
        out = out.view(out.size(0), 256 * 2 * 2)  # torch.flatten(out, 1)
        net['embeddings'] = out

        # classifier
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out = self.linear3(out)
        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        if self.field is None:
            return net
        else:
            return net[self.field]
