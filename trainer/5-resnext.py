# Hello world example for a ResNext network with external data which can not simply be loaded from the bucket.
# Transfers a h5py file from the bucket to the local machine, processes the data and
# trains for some epochs. No storing is done.
# The ResNext model in keras is taken from:
# https://github.com/titu1994/Keras-ResNeXt

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
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
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
    depth = 20
    cardinality = 8
    width = 32
    weight_decay = 5e-4

    print("input shape " + str(x_train.shape[1:]))
    # Create the dense net model without the top layer
    initial_model = ResNext(x_train.shape[1:], depth, cardinality, width, weight_decay)
    print("Inital model output shape " + str(initial_model.output_shape))
    # add the last layer
    last = initial_model.output

    x = GlobalMaxPooling2D()(last)

    #x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(initial_model.input, preds)

    #GlobalAveragePooling2D
    print("Model Output shape " + str(model.output_shape))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

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




# Copied from ResNext file to avoid import, nothing Google Cloud ML specific anymore
def ResNext(input_shape=None, depth=29, cardinality=8, width=64, weight_decay=5e-4,
                include_top=False, weights=None, input_tensor=None,
                pooling=None, classes=10):
        """Instantiate the ResNeXt architecture. Note that ,
            when using TensorFlow for best performance you should set
            `image_data_format="channels_last"` in your Keras config
            at ~/.keras/keras.json.
            The model are compatible with both
            TensorFlow and Theano. The dimension ordering
            convention used by the model is the one
            specified in your Keras config file.
            # Arguments
                depth: number or layers in the ResNeXt model. Can be an
                    integer or a list of integers.
                cardinality: the size of the set of transformations
                width: multiplier to the ResNeXt width (number of filters)
                weight_decay: weight decay (l2 norm)
                include_top: whether to include the fully-connected
                    layer at the top of the network.
                weights: `None` (random initialization)
                input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                    to use as image input for the model.
                input_shape: optional shape tuple, only to be specified
                    if `include_top` is False (otherwise the input shape
                    has to be `(32, 32, 3)` (with `tf` dim ordering)
                    or `(3, 32, 32)` (with `th` dim ordering).
                    It should have exactly 3 inputs channels,
                    and width and height should be no smaller than 8.
                    E.g. `(200, 200, 3)` would be one valid value.
                pooling: Optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional layer.
                    - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                        be applied.
                classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            # Returns
                A Keras model instance.
            """

        if weights not in {'cifar10', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `cifar10` '
                             '(pre-training on CIFAR-10).')

        if weights == 'cifar10' and include_top and classes != 10:
            raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
                             ' as true, `classes` should be 10')

        if type(depth) == int:
            if (depth - 2) % 9 != 0:
                raise ValueError('Depth of the network must be such that (depth - 2)'
                                 'should be divisible by 9.')

        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=32,
                                          min_size=8,
                                          data_format=K.image_data_format(),
                                          include_top=include_top)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = __create_res_next(classes, img_input, include_top, depth, cardinality, width,
                              weight_decay, pooling)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='resnext')

        # load weights
        if weights == 'cifar10':
            if (depth == 29) and (cardinality == 8) and (width == 64):
                # Default parameters match. Weights for this model exist:

                if K.image_data_format() == 'channels_first':
                    if include_top:
                        weights_path = get_file('resnext_cifar_10_8_64_th_dim_ordering_th_kernels.h5',
                                                CIFAR_TH_WEIGHTS_PATH,
                                                cache_subdir='models')
                    else:
                        weights_path = get_file('resnext_cifar_10_8_64_th_dim_ordering_th_kernels_no_top.h5',
                                                CIFAR_TH_WEIGHTS_PATH_NO_TOP,
                                                cache_subdir='models')

                    model.load_weights(weights_path)

                    if K.backend() == 'tensorflow':
                        warnings.warn('You are using the TensorFlow backend, yet you '
                                      'are using the Theano '
                                      'image dimension ordering convention '
                                      '(`image_dim_ordering="th"`). '
                                      'For best performance, set '
                                      '`image_dim_ordering="tf"` in '
                                      'your Keras config '
                                      'at ~/.keras/keras.json.')
                        convert_all_kernels_in_model(model)
                else:
                    if include_top:
                        weights_path = get_file('resnext_cifar_10_8_64_tf_dim_ordering_tf_kernels.h5',
                                                CIFAR_TF_WEIGHTS_PATH,
                                                cache_subdir='models')
                    else:
                        weights_path = get_file('resnext_cifar_10_8_64_tf_dim_ordering_tf_kernels_no_top.h5',
                                                CIFAR_TF_WEIGHTS_PATH_NO_TOP,
                                                cache_subdir='models')

                    model.load_weights(weights_path)

                    if K.backend() == 'theano':
                        convert_all_kernels_in_model(model)

        return model


def __initial_conv_block(input, weight_decay=5e-4):
    ''' Adds an initial convolution block, with batch normalization and relu activation
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x


def __initial_conv_block_inception(input, weight_decay=5e-4):
    ''' Adds an initial conv block, with batch norm and relu for the inception resnext
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(64, (7, 7), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), strides=(2, 2))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    return x


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    for c in range(cardinality):
        x = Conv2D(grouped_channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)

    return group_merge


def __bottleneck_block(input, filters=64, cardinality=8, width=4, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        width: widening factor
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input

    grouped_channels = int(filters * (width / 64))
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != filters * 4:
            init = Conv2D(filters * 4, (1, 1), activation='linear', padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != filters * 4:
            init = Conv2D(filters * 4, (1, 1), activation='linear', padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 4, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = Activation('relu')(x)

    return x


def __create_res_next(nb_classes, img_input, include_top, depth=29, cardinality=8, width=4,
                      weight_decay=5e-4, pooling=None):
    ''' Creates a ResNeXt model with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Can be an positive integer or a list
               Compute N = (n - 2) / 9.
               For a depth of 56, n = 56, N = (56 - 2) / 9 = 6
               For a depth of 101, n = 101, N = (101 - 2) / 9 = 11
        cardinality: the size of the set of transformations.
               Increasing cardinality improves classification accuracy,
        width: Width of the network.
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''

    if type(depth) is list or type(depth) is tuple:
        # If a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        # Otherwise, default to 3 blocks each of default number of group convolution blocks
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = 64
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters

    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, width, strides=1, weight_decay=weight_decay)

    N = N[1:]  # remove the first block from block definition list
    filters_list = filters_list[1:]  # remove the first filter from the filter list

    # block 2 to N
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, width, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, width, strides=1,
                                       weight_decay=weight_decay)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x

# Start the training for package
def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()