# Hello world example for a DenseNet network with external data which can not simply be loaded from the bucket.
# Transfers a h5py file from the bucket to the local machine, processes the data and
# trains for some epochs. No storing is done.
# DenseNet from https://github.com/liuzhuang13/DenseNet
#  Keras implementation from https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import os.path
import subprocess
import socket

import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



import warnings

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K


# Taking arguments from the command line
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')

batch_size = 32
num_classes = 5
epochs = 4
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

CIFAR_TH_WEIGHTS_PATH = ''
CIFAR_TF_WEIGHTS_PATH = ''
CIFAR_TH_WEIGHTS_PATH_NO_TOP = ''
CIFAR_TF_WEIGHTS_PATH_NO_TOP = ''

IMAGENET_TH_WEIGHTS_PATH = ''
IMAGENET_TF_WEIGHTS_PATH = ''
IMAGENET_TH_WEIGHTS_PATH_NO_TOP = ''
IMAGENET_TF_WEIGHTS_PATH_NO_TOP = ''


# Create validation set internally
def split_sets(X, Y, percentage):
    n = len(X)
    nb_samples = int(n * percentage)
    # sample without replacement from [0,n]
    samples = np.random.choice(n, nb_samples, replace=False)
    X_validation = [X[i] for i in samples]
    Y_validation = [Y[i] for i in samples]
    X = [X[i] for i in range(0, n) if i not in samples]
    Y = [Y[i] for i in range(0, n) if i not in samples]
    #
    return np.asarray(X), np.asarray(Y), np.asarray(X_validation), np.asarray(Y_validation)


# Copied the shuffle function from tflearn
def shuffle(*arrs):
    """ shuffle.

    Shuffle given arrays at unison, along first axis.

    Arguments:
        *arrs: Each array to shuffle at unison.

    Returns:
        Tuple of shuffled arrays.

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


# Where we actually do the work, loading and processing the data
def run_training():
    # Parsing the input argument(s)
    file_name = 'dataset_s.h5'
    #if socket.gethostname() == "heinzerm-Lenovo-Z51-70":
        #csv_file = os.path.join(FLAGS.input_dir, 'dataset_s.h5');
        # h5f = h5py.File(csv_file, 'r')
    #else:
    input_file = os.path.join(FLAGS.input_dir, file_name);
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', input_file, '/tmp'])
    h5f = h5py.File(os.path.join('/tmp', file_name), 'r')
    #
    X = h5f['X']
    Y = h5f['Y']
    X, Y = shuffle(X, Y)

    x_train, y_train, x_test, y_test = split_sets(X, Y, 0.2)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #parameter for resnext model
    img_dim = x_train.shape[1:]
    depth = 40 # L param
    nb_dense_block = 1
    growth_rate = 36 # k param
    nb_filter = 16
    dropout_rate = 0.2
    weight_decay = 1E-4
    learning_rate = 1E-4

    print("input shape " + str(x_train.shape[1:]))
    # Create the dense net model without the top layer
    model = DenseNet(num_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    print("Model Output shape " + str(model.output_shape))

    # initiate RMSprop optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=True,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))




# From DenseNet file to avoid import
def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 1, 1,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, nb_layers, nb_filter, growth_rate,
                      dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)
    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    * The main difference between this implementation and the implementation
    above is that the one above
    """

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        x = merge([merge_tensor, x], mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      name="initial_conv2D",
                      bias=False,
                      W_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(dim_ordering="th")(x)
    x = Dense(nb_classes,
              activation='softmax',
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    densenet = Model(input=[model_input], output=[x], name="DenseNet")

    return densenet

# Start the training for package
def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
