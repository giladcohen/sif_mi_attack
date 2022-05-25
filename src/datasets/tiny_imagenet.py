from PIL import Image
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from utils import get_all_files_recursive, convert_grayscale_to_rgb, inverse_map

class TinyImageNet(VisionDataset):
    """`TinyImageNet <https://www.kaggle.com/c/tiny-imagenet>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directories ``train``, ``val``, and ``test``
            exist or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    GLOVE_EMB_DIM = 300
    BERT_EMB_DIM = 1024

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.file_name = 'tiny-imagenet-200.zip'
        self.text_url = 'https://75acb9ce-c4ff-4914-a21d-3d50535914e4.usrfiles.com/archives/75acb9_ca1177a31a6143639b039a6e38d72b5a.zip'
        self.text_file_name = 'imagenet_text_embeddings.zip'

        self.base_dir = os.path.join(self.root, self.file_name[:-4].replace('-', '_'))
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.test_dir = os.path.join(self.base_dir, 'val')

        self.train_data_file = os.path.join(self.root, 'train_data.npy')
        self.train_labels_file = os.path.join(self.root, 'train_labels.npy')
        self.test_data_file = os.path.join(self.root, 'test_data.npy')
        self.test_labels_file = os.path.join(self.root, 'test_labels.npy')

        self.train = train  # training set or test set
        if download:
            self.download()

        self.class_to_idx = self.parse_classes()
        self.idx_to_class = inverse_map(self.class_to_idx)
        self.classes = self.get_class_names()

        self.data: Any = []
        self.targets = []

        if self.train:
            if os.path.exists(self.train_data_file):
                self.data = np.load(self.train_data_file)
                self.targets = np.load(self.train_labels_file)
            else:
                self.parse_train_data()
        else:
            if os.path.exists(self.test_data_file):
                self.data = np.load(self.test_data_file)
                self.targets = np.load(self.test_labels_file)
            else:
                self.parse_test_data()

    def get_class_names(self):
        class_key_to_desc = {}
        class_idx_to_desc = {}
        # classes = np.array(200, dtype=str)
        file = os.path.join(self.base_dir, 'words.txt')
        with open(file) as f:
            content = f.readlines()
        classes_keys = [x.split('\t')[0] for x in content]
        classes_desc = [x.split('\t')[1].split('\n')[0] for x in content]
        for i, key in enumerate(classes_keys):
            if key in self.class_to_idx.keys():
                class_key_to_desc[key] = classes_desc[i]
                class_idx_to_desc[self.class_to_idx[key]] = classes_desc[i]

        classes = []
        for i in range(200):
            classes.append(class_idx_to_desc[i])
        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        if not os.path.exists(self.base_dir):
            download_and_extract_archive(self.url, self.root, filename=self.file_name)
            os.rename(os.path.join(self.root, self.file_name[:-4]), self.base_dir)
            download_and_extract_archive(self.text_url, self.root, filename=self.text_file_name)

    def parse_classes(self):
        file = os.path.join(self.base_dir, 'wnids.txt')
        with open(file) as f:
            classes_ = f.readlines()
        classes_ = [x.strip() for x in classes_]
        class_map = {}
        for class_id, class_name in enumerate(classes_):
            class_map[class_name] = class_id
        return class_map

    def parse_train_data(self):
        """Parsing the training data in imagenet:
            .../tiny_imagenet/tiny_imagenet_200/train$ ll
            total 1M
            drwxrwxr-x   3 1M Jul 24 14:46 n03584254
            drwxrwxr-x   3 1M Jul 24 14:46 n02403003
            drwxrwxr-x   3 1M Jul 24 14:46 n02056570
        sum of 200 folders, every folder has 500 images in:
        .../tiny_imagenet/tiny_imagenet_200/train/n07749582/images$ ls | cat -n
             1	n07749582_0.JPEG
             2	n07749582_1.JPEG
             3	n07749582_10.JPEG
        """
        files = get_all_files_recursive(self.train_dir, 'JPEG')
        print('Parsing train data...')
        for file in tqdm(files):
            img = Image.open(file)
            np_img = np.asarray(img)
            if np_img.shape == (64, 64, 3):
                self.data.append(np_img)
            elif np_img.shape == (64, 64):
                self.data.append(convert_grayscale_to_rgb(np_img))
            else:
                raise AssertionError('illegal shape of {}'.format(np_img.shape))

            class_name = os.path.basename(file).split('_')[0]
            class_id = self.class_to_idx[class_name]
            self.targets.append(class_id)

        self.data = np.vstack(self.data).reshape(-1, 64, 64, 3)
        self.targets = np.asarray(self.targets)

        # save for quick access:
        np.save(os.path.join(self.root, 'train_data.npy'), self.data)
        np.save(os.path.join(self.root, 'train_labels.npy'), self.targets)

    def parse_test_data(self):
        """Parsing the validation data in imagenet:
            .../tiny_imagenet/tiny_imagenet_200/val ll
            total 1M
            -rw-rw-r-- 1 user user 1M Jul 24 14:47 val_annotations.txt
            drwxrwxr-x 2 user user 1M Jul 24 14:47 images

        images has 10000 images (200 x 50): 50 images of each of the 200 classes.
        -rw-rw-r-- 1 user user 1M Jul 24 14:47 val_5261.JPEG
        -rw-rw-r-- 1 user user 1M Jul 24 14:47 val_2584.JPEG
        -rw-rw-r-- 1 user user 1M Jul 24 14:47 val_1664.JPEG
        """
        self.data = np.empty((10000, 64, 64, 3), dtype=np.uint8)
        self.targets = -1 * np.empty(10000, dtype=int)

        labels_file = os.path.join(self.test_dir, 'val_annotations.txt')
        with open(labels_file) as f:
            content = f.readlines()
        img_id = list(map(int, [x.split('val_')[1].split('.JPEG')[0] for x in content]))
        img_classes = [x.split('.JPEG\t')[1].split('\t')[0] for x in content]
        assert len(img_id) == len(img_classes) == 10000
        for i in range(10000):
            self.targets[img_id[i]] = self.class_to_idx[img_classes[i]]
        assert (self.targets != -1).all(), 'not all targets had been set.'

        files = get_all_files_recursive(self.test_dir, 'JPEG')
        assert len(files) == 10000, 'There should be exactly 10000 images in the validation set'
        print('Parsing test data...')
        for file in tqdm(files):
            img = Image.open(file)
            np_img = np.asarray(img)
            img_id = int(os.path.basename(file).split('_')[1].split('.JPEG')[0])

            if np_img.shape == (64, 64, 3):
                self.data[img_id] = np_img
            elif np_img.shape == (64, 64):
                self.data[img_id] = convert_grayscale_to_rgb(np_img)
            else:
                raise AssertionError('illegal shape of {}'.format(np_img.shape))

        # save for quick access:
        np.save(os.path.join(self.root, 'test_data.npy'), self.data)
        np.save(os.path.join(self.root, 'test_labels.npy'), self.targets)
