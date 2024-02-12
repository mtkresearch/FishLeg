import torch
import torch.nn as nn
import numpy as np
import os
import gzip
import urllib.request

from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import copy
import argparse

__all__ = [
    "dense_to_one_hot",
    "maybe_download",
    "ImageDataSet",
    "extract_images",
    "extract_labels",
    "read_data_sets",
]


def dense_to_one_hot(y, max_value=9, min_value=0):
    """
    converts y into one hot reprsentation.
    Parameters
    ----------
    y : list
        A list containing continous integer values.
    Returns
    -------
    one_hot : numpy.ndarray
        A numpy.ndarray object, which is one-hot representation of y.
    """
    length = len(y)
    one_hot = torch.zeros((length, (max_value - min_value + 1)))
    for i in range(length):
        one_hot[i, y[i]] = 1
    return one_hot


def maybe_download(SOURCE_URL, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print("Succesfully downloaded", filename, statinfo.st_size, "bytes.")
    return filepath


class ImageDataSet(Dataset):
    def __init__(self, images, labels, if_autoencoder, input_reshape, if_faces):
        self._num_examples = len(images)
        if len(images) > 0:
            if input_reshape:
                images = images.reshape(
                    images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
                )
            images = images.astype(np.float32)
            if if_faces:
                images = np.swapaxes(images, 1, 2)
                images = images.reshape(images.shape[0], images.shape[1], 25, 25)
            if if_autoencoder:
                labels = images
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def sample(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return torch.from_numpy(self._images[start:end]).to(
            torch.float32
        ), torch.from_numpy(self._labels[start:end]).to(torch.float32)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def length(self):
        return self._num_examples

    @property
    def data(self):
        return self._images

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self._images[idx]), torch.tensor(self._labels[idx])


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                "Invalid magic number %d in MNIST image file: %s" % (magic, filename)
            )
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 1, rows, cols)
        return data


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                "Invalid magic number %d in MNIST label file: %s" % (magic, filename)
            )
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return

def get_MNIST(path, if_autoencoder=False):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data
    train_data = datasets.MNIST(
        root=path, train=True, download=True, transform=transform, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )

    # Download and load the test data
    test_data = datasets.MNIST(
        root=path, train=False, download=True, transform=transform, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )

    class DataSets(object):
        pass

    data_sets = DataSets()

    if if_autoencoder:
        train_data.targets = train_data.data
        test_data.targets = test_data.data

    data_sets.train = train_data
    data_sets.test = test_data

    return data_sets

def read_cifar(path, if_autoencoder=False):
    # code from https://github.com/jeonsworld/MLP-Mixer-Pytorch/blob/main/utils/data_utils.py
    image_size = 32
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )
    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )

    class DataSets(object):
        pass

    data_sets = DataSets()

    if if_autoencoder:
        train_data.targets = train_data.data
        test_data.targets = test_data.data

    data_sets.train = train_data
    data_sets.test = test_data

    return data_sets

def read_data_sets(name_dataset, home_path, if_autoencoder=True, reshape=True):
    """
    A helper utitlity that returns ImageDataset.
    If the data are not present in the home_path they are
    downloaded from the appropriate site.

    * Input*
    name_dataset: MNIST, FACES or CURVES
    home_path:    The root folder to look for or download the dataset.
    batch_size:   Batch size.

    *Returns*:
    An ImageDataset class object that implements get_batch().
    """

    class DataSets(object):
        pass

    data_sets = DataSets()
    if_faces = False

    VALIDATION_SIZE = 0
    train_dir = os.path.join(home_path, "data", name_dataset + "_data")

    print(f"Begin loading data for {name_dataset}")
    if name_dataset == "MNIST":
        if_autoencoder = if_autoencoder

        SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
        TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
        TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
        TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
        TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        print(f"Data read from {local_file}")
        train_images = extract_images(local_file)

        local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
        test_images = extract_images(local_file)

        local_file = maybe_download(SOURCE_URL, TRAIN_LABELS, train_dir)
        print(f"Data read from {local_file}")
        train_labels = extract_labels(local_file, one_hot=True)

        local_file = maybe_download(SOURCE_URL, TEST_LABELS, train_dir)
        test_labels = extract_labels(local_file, one_hot=True)

        # see "Reducing the Dimensionality of Data with Neural Networks"
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
    elif name_dataset == "FACES":
        if_autoencoder = if_autoencoder
        if_faces = True

        SOURCE_URL = "http://www.cs.toronto.edu/~jmartens/"
        TRAIN_IMAGES = "newfaces_rot_single.mat"

        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        print(f"Data read from {local_file}")

        numpy_file = os.path.dirname(local_file) + "/faces.npy"
        if os.path.exists(numpy_file):
            images_ = np.load(numpy_file)
        else:
            import mat4py

            images_ = mat4py.loadmat(local_file)
            images_ = np.asarray(images_["newfaces_single"])

            images_ = np.transpose(images_)
            np.save(numpy_file, images_)
            print(f"Data saved to {numpy_file}")

        train_images = images_[:103500]
        test_images = images_[103500:]

        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]

        train_labels = train_images
        test_labels = test_images

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]

    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    input_reshape = reshape

    data_sets.train = ImageDataSet(
        train_images, train_labels, if_autoencoder, input_reshape, if_faces
    )
    data_sets.validation = ImageDataSet(
        validation_images, validation_labels, if_autoencoder, input_reshape, if_faces
    )
    data_sets.test = ImageDataSet(
        test_images, test_labels, if_autoencoder, input_reshape, if_faces
    )

    print(f"Succesfully loaded {name_dataset} dataset.")
    return data_sets
