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
import math

torch.set_default_dtype(torch.float32)

from optim.FishLeg import FishLeg, FixedGaussianLikelihood, BernoulliLikelihood

def dense_to_one_hot(y, max_value=9,min_value=0):
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
    return  one_hot.at[list(range(length)), y].set(1)

def maybe_download(SOURCE_URL, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

class ImageDataSet(object):
    def __init__(self, images, labels, if_autoencoder, input_reshape):
        self._num_examples = len(images)
        if len(images)>0:
            if input_reshape == 'fully-connected':
                images = np.swapaxes(images, 2, 3)
                images = np.swapaxes(images, 1, 2)
                images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2] * images.shape[3])
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
        return torch.from_numpy(self._images[start:end]).to(torch.float32), \
                torch.from_numpy(self._labels[start:end]).to(torch.float32)

    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def length(self):
        return self._num_examples
    
    @property
    def data(self):
        return self._images

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

    
def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return 

def read_data_sets(name_dataset, home_path, if_autoencoder = True):
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
    train_dir = os.path.join(home_path, 'data', name_dataset + '_data')

    print(f'Begin loading data for {name_dataset}')
    if name_dataset == 'MNIST':
        if_autoencoder = if_autoencoder
        
        
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        print(f'Data read from {local_file}')
        train_images = extract_images(local_file)

        local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
        test_images = extract_images(local_file)
        
        local_file = maybe_download(SOURCE_URL, TRAIN_LABELS, train_dir)
        print(f'Data read from {local_file}')
        train_labels = extract_labels(local_file,one_hot=True)

        local_file = maybe_download(SOURCE_URL, TEST_LABELS, train_dir)
        test_labels = extract_labels(local_file,one_hot=True)

        # see "Reducing the Dimensionality of Data with Neural Networks"
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
    elif name_dataset == 'FACES':
        if_autoencoder = if_autoencoder
        
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'newfaces_rot_single.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        print(f'Data read from {local_file}')
        import mat4py
        images_ = mat4py.loadmat(local_file)
        images_ = np.asarray(images_['newfaces_single'])
        images_ = np.transpose(images_)

        train_images = images_[:103500]
        test_images = images_[-41400:]

        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels = train_images
        test_labels = test_images  
        
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    input_reshape = 'fully-connected'
    
    data_sets.train = ImageDataSet(train_images, train_labels, if_autoencoder, input_reshape)
    data_sets.validation = ImageDataSet(validation_images, validation_labels, if_autoencoder, input_reshape)
    data_sets.test = ImageDataSet(test_images, test_labels, if_autoencoder, input_reshape)

    print(f'Succesfull loaded {name_dataset} dataset.')    
    return data_sets

if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = None

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    ## Hyperparams
    batch_size = 100
    epochs = 10
    eta_adam = 1e-4
    eta_fl = 5e-2
    eta_sgd = 1e-2
    aux_eta = 1e-4
    weight_decay = 1e-5
    beta = 0.9
    damping = 1.0
    eps = 1e-4

    ## Dataset
    dataset = read_data_sets('FACES', '../data/', if_autoencoder=True)
    input_dist = dataset.train
    aux_dist = dataset.train

    test_dist = dataset.test
    D_test = test_dist.sample(10000)
    class_data = dataset.train.images
    
    likelihood = FixedGaussianLikelihood(sigma_fixed=1.0)

    def nll(model, data):
        data_x, data_y = data
        pred_y = model.forward(data_x)
        return likelihood.nll(data_y, pred_y)

    def draw(model, data_x):
        pred_y = model.forward(data_x)
        return (data_x, likelihood.draw(pred_y))

    def dataloader():
        data_x, _ = aux_dist.sample(batch_size)
        return data_x
    
    
    model = nn.Sequential(
            nn.Linear(625, 2000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(2000, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 30, dtype=torch.float32),
            nn.Linear(30, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 2000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(2000, 625, dtype=torch.float32)
        )

    print("lr fl={}, lr sgd={}".format(eta_fl, eta_sgd))

    opt = FishLeg(
                model,
                draw,
                likelihood.nll,
                dataloader,
                lr=eta_fl,
                eps=eps,
                beta=beta,
                weight_decay=1e-5,
                update_aux_every=10,
                aux_lr=aux_eta,
                aux_betas=(0.9, 0.999),
                aux_eps=1e-8,
                damping=damping,
                pre_aux_training=25,
                sgd_lr=eta_sgd
            )
    
    FL_time = []
    LOSS = []
    TEST_LOSS = []
    st = time.time()

    for e in range(1, epochs+1):
        print("######## EPOCH", e)
        for t, j in enumerate(range(0, len(class_data), batch_size)):    
            data_x, data_y = input_dist.sample(batch_size)
            opt.zero_grad()

            pred_y = opt.model(data_x)
            loss = likelihood.nll(data_y, pred_y)
            loss.backward(retain_graph=True)
            opt.step((data_x, pred_y))
            
            if t % 50 == 0:
                FL_time.append(time.time() - st)
                LOSS.append(loss.detach().numpy())
                test_loss = nll(opt.model, D_test).detach().numpy()
                TEST_LOSS.append(test_loss)

                print(t, loss.detach().numpy(), test_loss)
                
    
    plt.plot(FL_time, LOSS, label='eps={}'.format(eps)) #color=colors_group[i])
    plt.plot(FL_time, TEST_LOSS, label='eps={} test'.format(eps)) #linestyle='--', color=colors_group[i])
    
    
    model = nn.Sequential(
            nn.Linear(625, 2000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(2000, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 30, dtype=torch.float32),
            nn.Linear(30, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 2000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(2000, 625, dtype=torch.float32)
        )

    opt = optim.Adam(model.parameters(), 
                    lr=eta_adam, 
                    betas=(0.9, 0.999), 
                    weight_decay=weight_decay,
                    eps=1e-8)

    FL_time = []
    LOSS = []
    TEST_LOSS = []
    st = time.time()
    for e in range(1, epochs+1):
        print("######## EPOCH", e)
        for t, j in enumerate(range(0, len(class_data), batch_size)):    
            D = input_dist.sample(batch_size)
            opt.zero_grad()
            loss = nll(model, D)
            loss.backward(retain_graph=True)
            opt.step()
            
            if t % 50 == 0:
                FL_time.append(time.time() - st)
                LOSS.append(loss.detach().numpy())
                test_loss = nll(model, D_test).detach().numpy()
                TEST_LOSS.append(test_loss)

                print(t, loss.detach().numpy(), test_loss)

    plt.plot(FL_time, LOSS, label='Adam', color='blue')
    plt.plot(FL_time, TEST_LOSS, label='Adam_test', linestyle='--', color='blue')
    
    plt.legend()

    plt.savefig("result/result.png", dpi=300)
