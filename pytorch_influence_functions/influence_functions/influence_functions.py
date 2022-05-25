#! /usr/bin/env python3
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import time
import datetime
import numpy as np
import copy
import logging
from tqdm import tqdm

from pathlib import Path

from pytorch_influence_functions.influence_functions.hvp_grad import (
    grad_z,
    s_test_sample,
)
from pytorch_influence_functions.influence_functions.utils import (
    save_json,
    display_progress,
)

from src.consts import RGB_MEAN, RGB_STD
from src.datasets.my_vision_dataset import MyVisionDataset

def calc_s_test(
    model,
    test_loader,
    train_loader,
    save=False,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    start=0,
):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = s_test_sample(
            model, z_test, t_test, train_loader, gpu, damp, scale, recursion_depth, r
        )

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec, save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test")
            )
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i - start, len(test_loader.dataset) - start
        )

    return s_tests, save


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)
        display_progress("Calc. grad_z: ", i - start, len(train_loader.dataset) - start)

    return grad_zs, save_pth


def load_s_test(s_test_dir=Path("./s_test/"), test_dataset_size=10, suffix='recdep500_r1'):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.

    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated for
        test_dataset_size: int, number of s_tests vectors expected

    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole dataset.
        s_test: list of torch vectors, contain all s_test for the whole dataset. Can be huge."""
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = len(list(s_test_dir.glob("*.s_test")))
    if num_s_test_files != test_dataset_size:
        logging.warning(
            "Load Influence Data: number of s_test sample files"
            " mismatches the available samples"
        )
    for i in range(num_s_test_files):
        s_test.append(torch.load(os.path.join(s_test_dir, str(i) + '_' + suffix + '.s_test')))
        display_progress("s_test files loaded: ", i, test_dataset_size)

    return s_test


def calc_all_influences(grad_z_dir, train_dataset_size, s_test_dir, test_dataset_size):
    grad_z_vecs = load_grad_z(grad_z_dir=grad_z_dir, train_dataset_size=train_dataset_size)
    suffix = 'recdep{}_r1'.format(train_dataset_size)
    s_test_vecs = load_s_test(s_test_dir=s_test_dir, test_dataset_size=test_dataset_size, suffix=suffix)

    influences = torch.zeros(test_dataset_size, train_dataset_size)
    for i in tqdm(range(test_dataset_size)):
        s_test_vec = s_test_vecs[i]
        for j in range(train_dataset_size):
            grad_z_vec = grad_z_vecs[j]
            with torch.no_grad():
                tmp_influence = (
                        -sum(
                            [
                                torch.sum(k * j).data
                                for k, j in zip(grad_z_vec, s_test_vec)
                            ]
                        )
                        / train_dataset_size
                )
            influences[i, j] = tmp_influence
    influences = influences.cpu().numpy()
    return influences


def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.

    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = len(list(grad_z_dir.glob("*.grad_z")))
    if available_grad_z_files != train_dataset_size:
        logging.warning("Load Influence Data: number of grad_z files mismatches" " the dataset size")
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        grad_z_vecs.append(torch.load(os.path.join(grad_z_dir, str(i) + '.grad_z')))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs


def calc_influence_function(train_dataset_size, grad_z_vecs=None, e_s_test=None):
    """Calculates the influence function

    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z()
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)

    if len(grad_z_vecs) != train_dataset_size:
        logging.warning("Training data size and the number of grad_z files are" " inconsistent.")
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = (
            -sum(
                [
                    ###################################
                    # TODO: verify if computation really needs to be done
                    # on the CPU or if GPU would work, too
                    ###################################
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(grad_z_vecs[i], e_s_test)
                    ###################################
                    # Originally with [i] because each grad_z contained
                    # a list of tensors as long as e_s_test list
                    # There is one grad_z per training data sample
                    ###################################
                ]
            )
            / train_dataset_size
        )
        influences.append(tmp_influence)
        # display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()

def calc_self_influence_average(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    for i in range(X.shape[0]):
        train_transform_gen = MyVisionDataset(X[i], y[i], transform=train_transform)  # just for transformations
        test_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X[i], 0)),
                                     torch.from_numpy(np.expand_dims(y[i], 0)))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        influences_tmp = []
        for k in range(8):
            X_aug, y_aug = train_transform_gen.__getitem__(0)
            train_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_aug, 0)),
                                          torch.from_numpy(np.expand_dims(y_aug, 0)))
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                      pin_memory=False, drop_last=False)
            influence, _, _, _ = calc_influence_single(net, train_loader, test_loader, 0, 0, rec_dep, r)
            influences_tmp.append(influence.item())

        influences.append(np.mean(influences_tmp))
    return np.asarray(influences)

def calc_self_influence_average_for_ref(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    test_transform = transforms.Normalize(RGB_MEAN, RGB_STD)
    for i in range(X.shape[0]):
        train_transform_gen = MyVisionDataset(X[i], y[i], transform=train_transform)  # just for transformations
        X_tensor = torch.tensor(X[i])
        X_transformed = test_transform(X_tensor).cpu().numpy()
        test_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_transformed, 0)),
                                     torch.from_numpy(np.expand_dims(y[i], 0)))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        influences_tmp = []
        for k in range(8):
            X_aug, y_aug = train_transform_gen.__getitem__(0)
            train_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_aug, 0)),
                                          torch.from_numpy(np.expand_dims(y_aug, 0)))
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                      pin_memory=False, drop_last=False)
            influence, _, _, _ = calc_influence_single(net, train_loader, test_loader, 0, 0, rec_dep, r)
            influences_tmp.append(influence.item())

        influences.append(np.mean(influences_tmp))
    return np.asarray(influences)

def calc_self_influence_adaptive(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    for i in range(X.shape[0]):
        train_dataset = MyVisionDataset(X[i], y[i], transform=train_transform)
        test_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X[i], 0)),
                                     torch.from_numpy(np.expand_dims(y[i], 0)))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        influence, _, _, _ = calc_influence_single_adaptive(net, train_loader, test_loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)

def calc_self_influence_adaptive_for_ref(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])

    for i in range(X.shape[0]):
        train_dataset = MyVisionDataset(X[i], y[i], transform=train_transform)
        test_dataset = MyVisionDataset(X[i], y[i], transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        influence, _, _, _ = calc_influence_single_adaptive(net, train_loader, test_loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)

def calc_self_influence(X, y, net, rec_dep, r):
    influences = []
    for i in range(X.shape[0]):
        tensor_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X[i], 0)),
                                       torch.from_numpy(np.expand_dims(y[i], 0)))
        loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
                            pin_memory=False, drop_last=False)
        influence, _, _, _ = calc_influence_single(net, loader, loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)

def calc_self_influence_for_ref(X, y, net, rec_dep, r):
    influences = []
    transform = transforms.Normalize(RGB_MEAN, RGB_STD)
    for i in range(X.shape[0]):
        X_tensor = torch.tensor(X[i])
        X_transformed = transform(X_tensor).cpu().numpy()
        tensor_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_transformed, 0)),
                                       torch.from_numpy(np.expand_dims(y[i], 0)))
        loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
                            pin_memory=False, drop_last=False)
        influence, _, _, _ = calc_influence_single(net, loader, loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)

def calc_single_influences(X_train, y_train, test_loader, net):
    train_size = X_train.shape[0]
    test_size = test_loader.dataset.__len__()
    influence_mat = np.zeros((test_size, train_size))
    for i in tqdm(range(test_size)):
        for j in range(train_size):
            tensor_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_train[j], 0)),
                                           torch.from_numpy(np.expand_dims(y_train[j], 0)))
            train_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
                                      pin_memory=False, drop_last=False)
            influence, _, _, _ = calc_influence_single(net, train_loader, test_loader, i, 0, 1, 1)
            influence_mat[i, j] = influence.item()
    return influence_mat

def calc_influence_single(
    model,
    train_loader,
    test_loader,
    test_id_num,
    gpu,
    recursion_depth,
    r,
    damp=0.01,
    scale=25,
    s_test_vec=None,
    time_logging=False,
    loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = s_test_sample(
            model,
            z_test,
            t_test,
            train_loader,
            gpu,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        if time_logging:
            time_a = datetime.datetime.now()

        grad_z_vec = grad_z(z, t, model, gpu=gpu)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                -sum(
                    [
                        ####################
                        # TODO: potential bottle neck, takes 17% execution time
                        # torch.sum(k * j).data.cpu().numpy()
                        ####################
                        torch.sum(k * j).data
                        for k, j in zip(grad_z_vec, s_test_vec)
                    ]
                )
                / train_dataset_size
            )

        influences.append(tmp_influence)

    influences = torch.stack(influences)
    influences = influences.cpu().numpy()
    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num


def calc_influence_single_adaptive(
    model,
    train_loader,
    test_loader,
    test_id_num,
    gpu,
    recursion_depth,
    r,
    damp=0.01,
    scale=25,
    s_test_vec=None,
    time_logging=False,
    loss_func="cross_entropy",
):
    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        # debug
        # z_img = convert_tensor_to_image(z_test[0].cpu().numpy())
        # plt.imshow(z_img)
        # plt.show()

        s_test_vec = s_test_sample(
            model,
            z_test,
            t_test,
            train_loader,
            gpu,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    assert train_dataset_size == 1
    influences = []
    num_layers = sum(1 for x in model.parameters())
    num_iters = 128
    grad_z_vec = []

    if time_logging:
        time_a = datetime.datetime.now()

    for i in range(num_iters):
        z, t = train_loader.dataset[0]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        # debug
        # z_img = convert_tensor_to_image(z[0].cpu().numpy())
        # plt.imshow(z_img)
        # plt.show()

        grad_z_vec_tmp = list(grad_z(z, t, model, gpu=gpu))
        for j in range(num_layers):
            if i == 0:
                grad_z_vec.append(torch.zeros_like(grad_z_vec_tmp[j]))
            else:
                grad_z_vec[j] += grad_z_vec_tmp[j]

    for j in range(num_layers):
        grad_z_vec[j] /= num_iters

    if time_logging:
        time_b = datetime.datetime.now()
        time_delta = time_b - time_a
        logging.info(
            f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
        )
    with torch.no_grad():
        tmp_influence = (
            -sum(
                [
                    ####################
                    # TODO: potential bottle neck, takes 17% execution time
                    # torch.sum(k * j).data.cpu().numpy()
                    ####################
                    torch.sum(k * j).data
                    for k, j in zip(grad_z_vec, s_test_vec)
                ]
            )
            / train_dataset_size
        )

    influences.append(tmp_influence)

    influences = torch.stack(influences)
    influences = influences.cpu().numpy()
    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num


def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader, start_index=0):
    """Gets the first num_samples from class class_id starting from
    start_index. Returns a list with the indicies which can be passed to
    test_loader.dataset[X] to retreive the actual data.

    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        test_loader: DataLoader, can load the test dataset.
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    sample_list = []
    img_count = 0
    for i in range(len(test_loader.dataset)):
        _, t = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and (img_count <= start_index + num_samples):
                sample_list.append(i)
            elif img_count > start_index + num_samples:
                break

    return sample_list


def get_dataset_sample_ids(num_samples, test_loader, num_classes=None, start_index=0):
    """Gets the first num_sample indices of all classes starting from
    start_index per class. Returns a list and a dict containing the indicies.

    Arguments:
        num_samples: int, number of samples of each class to return
        test_loader: DataLoader, can load the test dataset
        num_classes: int, number of classes contained in the dataset
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    sample_list = []
    if not num_classes:
        num_classes = len(np.unique(test_loader.dataset.targets))
    for i in range(num_classes):
        sample_dict[str(i)] = get_dataset_sample_ids_per_class(
            i, num_samples, test_loader, start_index
        )
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list) : len(sample_list)] = sample_dict[str(i)]
    return sample_dict, sample_list


def calc_img_wise(config, model, train_loader, test_loader, loss_func="cross_entropy"):
    """Calculates the influence function one test point at a time. Calcualtes
    the `s_test` and `grad_z` values on the fly and discards them afterwards.

    Arguments:
        config: dict, contains the configuration from cli params"""
    influences_meta = copy.deepcopy(config)
    test_sample_num = config["test_sample_num"]
    test_start_index = config["test_start_index"]
    outdir = Path(config["outdir"])

    # If calculating the influence for a subset of the whole dataset,
    # calculate it evenly for the same number of samples from all classes.
    # `test_start_index` is `False` when it hasn't been set by the user. It can
    # also be set to `0`.
    if test_sample_num and test_start_index is not False:
        test_dataset_iter_len = test_sample_num * config["num_classes"]
        _, sample_list = get_dataset_sample_ids(
            test_sample_num, test_loader, config["num_classes"], test_start_index
        )
    else:
        test_dataset_iter_len = len(test_loader.dataset)

    # Set up logging and save the metadata conf file
    logging.info(f"Running on: {test_sample_num} images per class.")
    logging.info(f"Starting at img number: {test_start_index} per class.")
    influences_meta["test_sample_index_list"] = sample_list
    influences_meta_fn = (
        f"influences_results_meta_{test_start_index}-" f"{test_sample_num}.json"
    )
    influences_meta_path = outdir.joinpath(influences_meta_fn)
    save_json(influences_meta, influences_meta_path)

    influences = {}
    # Main loop for calculating the influence function one test sample per
    # iteration.
    for j in range(test_dataset_iter_len):
        # If we calculate evenly per class, choose the test img indicies
        # from the sample_list instead
        if test_sample_num and test_start_index:
            if j >= len(sample_list):
                logging.warning(
                    "ERROR: the test sample id is out of index of the"
                    " defined test set. Jumping to next test sample."
                )
            i = sample_list[j]
        else:
            i = j

        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            model,
            train_loader,
            test_loader,
            test_id_num=i,
            gpu=config["gpu"],
            recursion_depth=config["recursion_depth"],
            r=config["r_averaging"],
            loss_func=loss_func,
        )
        end_time = time.time()

        ###########
        # Different from `influence` above
        ###########
        influences[str(i)] = {}
        _, label = test_loader.dataset[i]
        influences[str(i)]["label"] = label
        influences[str(i)]["num_in_dataset"] = j
        influences[str(i)]["time_calc_influence_s"] = end_time - start_time
        infl = [x.tolist() for x in influence]
        influences[str(i)]["influence"] = infl
        influences[str(i)]["harmful"] = harmful[:500]
        influences[str(i)]["helpful"] = helpful[:500]

        tmp_influences_path = outdir.joinpath(
            f"influence_results_tmp_"
            f"{test_start_index}_"
            f"{test_sample_num}"
            f"_last-i_{i}.json"
        )
        save_json(influences, tmp_influences_path)
        display_progress("Test samples processed: ", j, test_dataset_iter_len)

    logging.info(f"The results for this run are:")
    logging.info("Influences: ")
    logging.info(influence[:3])
    logging.info("Most harmful img IDs: ")
    logging.info(harmful[:3])
    logging.info("Most helpful img IDs: ")
    logging.info(helpful[:3])

    influences_path = outdir.joinpath(
        f"influence_results_{test_start_index}_" f"{test_sample_num}.json"
    )
    save_json(influences, influences_path)


def calc_all_grad_then_test(config, model, train_loader, test_loader):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    outdir = Path(config["outdir"])
    s_test_outdir = outdir.joinpath("s_test/")
    if not s_test_outdir.exists():
        s_test_outdir.mkdir()
    grad_z_outdir = outdir.joinpath("grad_z/")
    if not grad_z_outdir.exists():
        grad_z_outdir.mkdir()

    influence_results = {}

    calc_s_test(
        model,
        test_loader,
        train_loader,
        s_test_outdir,
        config["gpu"],
        config["damp"],
        config["scale"],
        config["recursion_depth"],
        config["r_averaging"],
        config["test_start_index"],
    )
    calc_grad_z(
        model, train_loader, grad_z_outdir, config["gpu"], config["test_start_index"]
    )

    train_dataset_len = len(train_loader.dataset)
    influences, harmful, helpful = calc_influence_function(train_dataset_len)

    influence_results["influences"] = influences
    influence_results["harmful"] = harmful
    influence_results["helpful"] = helpful
    influences_path = outdir.joinpath("influence_results.json")
    save_json(influence_results, influences_path)
