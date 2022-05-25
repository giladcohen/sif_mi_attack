from typing import Any, Tuple
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image
from utils import convert_tensor_to_image

class MyVisionDataset(VisionDataset):

    def __init__(self, data: np.ndarray, y_gt: np.ndarray, *args, **kwargs) -> None:
        root = None
        super().__init__(root, *args, **kwargs)
        assert isinstance(data, np.ndarray), 'type of data must be np.ndarray type, but got {} instead'.format(type(data))
        assert isinstance(y_gt, (np.int32, np.int64)), 'type of y_gt must be np.int type, but got {} instead'.format(type(y_gt))
        self.data = np.expand_dims(convert_tensor_to_image(data), 0)
        self.y_gt = np.expand_dims(y_gt, 0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        assert index == 0
        img, y_gt = self.data[index], self.y_gt[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            y_gt = self.target_transform(y_gt)

        return img, y_gt
