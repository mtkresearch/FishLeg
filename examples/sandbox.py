import torch
import torch.nn as nn
import numpy as np
import time
import os
import gzip
import urllib.request
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import copy
import argparse

torch.set_default_dtype(torch.float32)

from optim.FishLeg import FishLeg, FISH_LIKELIHOODS


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
    one_hot = jnp.zeros((length, (max_value - min_value + 1)))
    return one_hot.at[list(range(length)), y].set(1)


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
    def __init__(self, images, labels, if_autoencoder, input_reshape):
        self._num_examples = len(images)
        if len(images) > 0:
            if input_reshape == "fully-connected":
                images = np.swapaxes(images, 2, 3)
                images = np.swapaxes(images, 1, 2)
                images = images.reshape(
                    images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
                )
            images = images.astype(np.float32)
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
        data = data.reshape(num_images, rows, cols, 1)
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


def read_data_sets(name_dataset, home_path, if_autoencoder=True):
    """A helper utitlity that returns ImageDataset.
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

    input_reshape = "fully-connected"

    data_sets.train = ImageDataSet(
        train_images, train_labels, if_autoencoder, input_reshape
    )
    data_sets.validation = ImageDataSet(
        validation_images, validation_labels, if_autoencoder, input_reshape
    )
    data_sets.test = ImageDataSet(
        test_images, test_labels, if_autoencoder, input_reshape
    )

    print(f"Succesfully loaded {name_dataset} dataset.")
    return data_sets


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--exp", type=str, help="which dataset", default="MNIST")
    args = argparser.parse_args()

    seed = 13
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = None

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    ## Hyperparams
    if args.exp == "FACES":

        batch_size = 100
        epochs = 5
        eta_adam = 3e-5
        eta_fl = 0.05
        eta_sgd = 0.01
        aux_eta = 5e-4
        weight_decay = 1e-5
        beta = 0.9
        damping = 1.0

        dataset = read_data_sets("FACES", "../data/", if_autoencoder=True)

    if args.exp == "MNIST":
        batch_size = 100
        epochs = 10
        
        eta_adam = 1e-4

        fish_lr = 0.02
        beta = 0.9
        weight_decay = 1e-5
        update_aux_every = 10
        aux_lr = 2e-3
        aux_eps = 1e-8
        damping = 0.3
        pre_aux_training = 10
        scale = 1
        initialization = "normal"
        normalization = True
        batch_speedup = False
        fine_tune = False
        warmup = 0

        dataset = read_data_sets("MNIST", "../data/", if_autoencoder=True)

    ## Dataset
    train_dataset = dataset.train
    test_dataset = dataset.test
    if args.exp == "FACES":
        likelihood = FISH_LIKELIHOODS["fixedgaussian"](sigma=1.0, device=device)

        def mse(model, data):
            data_x, data_y = data
            pred_y = model.forward(data_x)
            return torch.mean(torch.square(pred_y - data_y))

    if args.exp == "MNIST":
        likelihood = FISH_LIKELIHOODS["bernoulli"](device=device)

        def mse(model, data):
            data_x, data_y = data
            pred_y = model.forward(data_x)
            pred_y = torch.sigmoid(pred_y)
            return torch.mean(torch.square(pred_y - data_y))

    def nll(model, data):
        data_x, data_y = data
        pred_y = model.forward(data_x)
        return likelihood.nll(data_y, pred_y)

    def draw(model, data):
        data_x, data_y = data
        pred_y = model.forward(data_x)
        return (data_x, likelihood.draw(pred_y))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    aux_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    test_loader_adam = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    if args.exp == "FACES":
        model = nn.Sequential(
            nn.Linear(625, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 30),
            nn.Linear(30, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 625),
        ).to(device)

    if args.exp == "MNIST":
        model = nn.Sequential(
            nn.Linear(784, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 250, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(250, 30, dtype=torch.float32),
            nn.Linear(30, 250, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(250, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 784, dtype=torch.float32),
        ).to(device)

    model_adam = copy.deepcopy(model)

    # print("lr fl={}, lr sgd={}, lr aux={}".format(eta_fl, eta_sgd, aux_eta))

    opt = FishLeg(
        model,
        draw,
        nll,
        aux_loader,
        likelihood,
        fish_lr=fish_lr,
        beta=beta,
        weight_decay=weight_decay,
        update_aux_every=update_aux_every,
        aux_lr=aux_lr,
        aux_betas=(0.9, 0.999),
        aux_eps=aux_eps,
        damping=damping,
        pre_aux_training=pre_aux_training,
        initialization=initialization,
        device=device,
        batch_speedup=batch_speedup,
        scale=scale,
    )

    print(opt.__dict__["fish_lr"])
    print(opt.__dict__["beta"])
    print(opt.__dict__["aux_lr"])
    print(opt.__dict__["damping"])
    print(opt.__dict__["scale"])

    FL_time = []
    LOSS = []
    AUX_LOSS = []
    TEST_LOSS = []
    st = time.time()
    iteration = 0
    for e in range(1, epochs + 1):
        print("######## EPOCH", e)
        for n, (batch_data, batch_labels) in enumerate(train_loader, start=1):
            iteration += 1
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            opt.zero_grad()
            loss = nll(opt.model, (batch_data, batch_labels))
            loss.backward()
            opt.step()

            if n % 50 == 0:
                FL_time.append(time.time() - st)
                LOSS.append(loss.detach().cpu().numpy())
                AUX_LOSS.append(opt.aux_loss)

                test_batch_data, test_batch_labels = next(iter(test_loader))
                test_batch_data, test_batch_labels = test_batch_data.to(
                    device
                ), test_batch_labels.to(device)
                test_loss = mse(opt.model, (test_batch_data, test_batch_labels))

                TEST_LOSS.append(test_loss.detach().cpu().numpy())

                print(n, LOSS[-1], AUX_LOSS[-1], TEST_LOSS[-1])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(FL_time, LOSS, label="Fishleg")  # color=colors_group[i])
    axs[1].plot(
        FL_time, TEST_LOSS, label="Fishleg"
    )  # linestyle='--', color=colors_group[i])

    opt = optim.Adam(
        model_adam.parameters(),
        lr=eta_adam,
        betas=(0.9, 0.9),
        weight_decay=weight_decay,
        eps=1e-4,
    )
    iteration = 0
    FL_time = []
    LOSS = []
    TEST_LOSS = []
    st = time.time()
    for e in range(1, epochs + 1):
        print("######## EPOCH", e)
        for n, (batch_data, batch_labels) in enumerate(train_loader):
            iteration += 1
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            opt.zero_grad()
            loss = nll(model_adam, (batch_data, batch_labels))
            loss.backward()
            opt.step()

            if n % 50 == 0:
                FL_time.append(time.time() - st)
                LOSS.append(loss.detach().cpu().numpy())
                test_batch_data, test_batch_labels = next(iter(test_loader_adam))
                test_batch_data, test_batch_labels = test_batch_data.to(
                    device
                ), test_batch_labels.to(device)
                test_loss = mse(model_adam, (test_batch_data, test_batch_labels))
                TEST_LOSS.append(test_loss.detach().cpu().numpy())

                print(n, LOSS[-1], TEST_LOSS[-1])

    axs[0].plot(FL_time, LOSS, label="Adam")
    axs[1].plot(FL_time, TEST_LOSS, label="Adam")

    axs[0].legend()
    axs[1].legend()

    axs[0].set_title("Training Loss")
    axs[1].set_title("Test MSE")
    fig.savefig("result/result.png", dpi=300)
