"""
MNIST handwritten digits dataset.

"""

import numpy as np
import os
import wget

class MNIST():

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.num_train = 0
        self.num_test = 0
        self.num_val = 0

    def load(self, path='data/mnist.npz'):
        """Loads the MNIST dataset.

        # Arguments
            path: path where to cache the dataset locally

        # Returns
            none
        """
        if not os.path.exists(path):
            print('start download mnist dataset...')
            wget.download('https://s3.amazonaws.com/img-datasets/mnist.npz', out=path)
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()

        # normalization
        # x_train = (x_train-np.mean(x_train, axis=0, keepdims=True))/255
        # x_test = (x_test-np.mean(x_test, axis=0, keepdims=True))/255

        x_train = x_train/255
        x_test = x_test/255

        x_train_shape = x_train.shape
        x_test_shape = x_test.shape
        x_train = x_train.reshape(x_train_shape[0], 1, x_train_shape[1], x_train_shape[2])
        x_test = x_test.reshape(x_test_shape[0], 1, x_test_shape[1], x_test_shape[2])

        self.num_train = int(x_train.shape[0] * 0.8)
        self.num_val = x_train.shape[0] - self.num_train
        self.num_test = x_test.shape[0]

        self.x_train = x_train[:self.num_train]
        self.y_train = y_train[:self.num_train]
        self.x_val = x_train[self.num_train:]
        self.y_val = y_train[self.num_train:]
        self.x_test = x_test
        self.y_test = y_test

        print('Number of training images: ', self.num_train)
        print('Number of validation images: ', self.num_val)
        print('Number of testing images: ', self.num_test)

    def train_loader(self, batch, shuffle=True):
        pointer = 0
        while True:
            if shuffle:
                idx = np.random.choice(self.num_train, batch)
            else:
                if pointer + batch <= self.num_train:
                    idx = np.arange(pointer, pointer+batch)
                    pointer = pointer + batch
                else:
                    pointer = 0
                    idx = np.arange(pointer, pointer+batch)
                    pointer = pointer + batch
            yield self.x_train[idx], self.y_train[idx]
    
    def test_loader(self, batch):
        pointer = 0
        while pointer+batch<=self.num_test:
            idx = np.arange(pointer, pointer+batch)
            pointer = pointer + batch
            yield self.x_test[idx], self.y_test[idx]
        if pointer<self.num_test-1:
            idx = np.arange(pointer, self.num_test-pointer-1)
            pointer = self.num_test-1
            yield self.x_test[idx], self.y_test[idx]
        else:
            return None

    def val_loader(self, batch):
        pointer = 0
        while pointer+batch<=self.num_val:
            idx = np.arange(pointer, pointer+batch)
            pointer = pointer + batch
            yield self.x_val[idx], self.y_val[idx]
        if pointer<self.num_val-1:
            idx = np.arange(pointer, self.num_val-pointer-1)
            pointer = self.num_val-1
            yield self.x_val[idx], self.y_val[idx]
        else:
            return None