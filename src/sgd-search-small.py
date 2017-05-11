# coding: utf-8

from __future__ import division

import os
import operator
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

from utils import *
from skimage import transform
from itertools import product
from tqdm import tqdm, tqdm_notebook
from keras import initializers
from keras.layers import Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop, Nadam

# share GPU memory
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.49
set_session(tf.Session(config=config))

### LOADING & PREPROCESS DATA
# data constants
max_size = (32, 32, 1)
data_files = {'cifar': '../data/cifar.npy',
              'mnist': '../data/mnist.npy',
              'lfw': '../data/lfw.npy'}

# load and transform data
data = np.load(data_files['mnist'])
data = data.astype('float32')

# shuffle data
np.random.shuffle(data)

# limit dataset
data = data[:40000]

if data.shape[1:3] > max_size:
    data = np.array([transform.resize(image, max_size, preserve_range=True, order=0)]
                    for image in data)

# get data sizes
data_count = data.shape[0]
data_size = data.shape[1:4] if data.shape[3] > 1 else data.shape[1:3]
data_dim = reduce(operator.mul, data.shape[1:])

# reshape data if necessary
data = data.reshape(data_count, *data_size)
print 'Loaded data {}'.format(data.shape)

# standardize data
data_reshaped = data.reshape(data_count, -1)
data_mean = np.mean(data_reshaped, axis=1, keepdims=True, dtype=np.float64)
data_z_mean = (data_reshaped - data_mean)
data_std = np.std(data_z_mean, axis=1, keepdims=True, dtype=np.float64) + 1e-16
data_normed = data_z_mean / data_std
data_proc = data_normed.reshape(data_count, *data_size)
print 'Before preproc', data.mean(), data.std()
print 'After preproc', data_proc.mean(), data_proc.std()


### MODEL FUNCTIONS
# latent space generators
def get_uniform_space(high, low, space_size):
    return lambda batch_size: np.random.uniform(low, high, (batch_size, space_size)).astype('float32')

def get_gaussian_space(mean, var, space_size):
    return lambda batch_size: np.random.normal(mean, var, (batch_size, space_size)).astype('float32')

def get_mlp_model(data_dim, latent_dim, sgd_lr=0.001, sgd_mom=0.1, gen_activation='tanh', relu_alpha=0.01, do_p=0.05):
    # setup optimizer
    opt = SGD(lr=sgd_lr, momentum=sgd_mom)

    leaky_alpha = relu_alpha
    dropout_p = do_p
    # setup generator network
    generator = Sequential()
    generator.add(Dense(128, input_dim=latent_dim))
    generator.add(LeakyReLU(alpha=leaky_alpha))
    generator.add(Dense(256))
    generator.add(LeakyReLU(alpha=leaky_alpha))
    generator.add(Dense(512))
    generator.add(LeakyReLU(alpha=leaky_alpha))
    generator.add(Dense(data_dim, activation=gen_activation))
    generator.compile(loss='binary_crossentropy', optimizer=opt)

    # setup discriminator network
    discriminator = Sequential()
    discriminator.add(Dense(512, input_dim=data_dim))
    discriminator.add(LeakyReLU(alpha=leaky_alpha))
    discriminator.add(Dropout(dropout_p))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(alpha=leaky_alpha))
    discriminator.add(Dropout(dropout_p))
    discriminator.add(Dense(128))
    discriminator.add(LeakyReLU(alpha=leaky_alpha))
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha=leaky_alpha))
    discriminator.add(Dropout(dropout_p))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)

    # setup combined network
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=opt)
    return (generator, discriminator, gan)

### GRID SEARCH
# setup experiment
max_epochs = 25
batch_size = 128

latent_dims = [100, 10000]
sgd_lrs=[0.0001, 0.001, 0.01]
sgd_moms=[0.1, 0.3]
gen_activations=['tanh', 'linear']
relu_alphas=[0.0, 0.05]
do_ps=[0.0, 0.3]

for latent_dim, sgd_lr, sgd_mom, gen_activation, relu_alpha, do_p in product(latent_dims, sgd_lrs, sgd_moms, gen_activations, relu_alphas, do_ps):
    plot_suffix = 'lat{}-sgdlr{}-sgdmom{}-gact{}-ralpha{}-dop{}'.format(latent_dim,
    str(sgd_lr).replace('.', '_'),
    str(sgd_mom).replace('.', '_'),
    gen_activation,
    str(relu_alpha).replace('.', '_'),
    str(do_p).replace('.', '_')
    )

    z_space = get_gaussian_space(0, 1, latent_dim)
    generator, discriminator, gan = get_mlp_model(data_dim, latent_dim)

    # prepare data
    dis_labels = np.ones(2*batch_size)
    dis_labels[:batch_size] = 0
    gan_labels = np.ones(batch_size)

    g_losses = []
    d_losses = []

    for epoch_ix in xrange(max_epochs):
        np.random.shuffle(data_proc)
        for batch_ix in tqdm(xrange(0, data_count-batch_size, batch_size)):
            # train discriminator
            discriminator.trainable = True
            z_samples = z_space(batch_size).astype('float32')
            x_samples = data_proc[batch_ix: batch_ix+batch_size].reshape(batch_size, -1)
            d_loss = discriminator.train_on_batch(
                np.vstack([generator.predict(z_samples), x_samples]), dis_labels)

            # train generator
            z_samples = z_space(batch_size).astype('float32')
            discriminator.trainable = False
            g_loss = gan.train_on_batch(z_samples, gan_labels)

        g_losses.append(g_loss)
        d_losses.append(d_loss)

        print 'epoch: {} -- loss G: {} - D: {}'.format(epoch_ix, g_loss, d_loss)

    perf_plot = plot_performance(('g_loss', g_losses), ('d_loss', d_losses))
    perf_plot.savefig('experiments/small_perf' + plot_suffix + '.png', dpi=160)

    z_samples = z_space(batch_size).astype('float32')
    fakes = generator.predict(z_samples[:8])
    sample_plot = plot_images(fakes, data_size);
    sample_plot.savefig('experiments/small_sample' + plot_suffix + '.png', dpi=160)
