# Hello world example for a CNN with external data which can not simply be loaded from the bucket.
# Transfers a h5py file from the bucket to the local machine, processes the data, and
# stores the model back on the cloud.
# The code for the CNN is from the Keras tutorial (cifar10_cnn):
# https://github.com/fchollet/keras/tree/master/examples

from __future__ import print_function, absolute_import


import tensorflow as tf
import numpy as np
import os.path
import subprocess

import keras
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# Taking arguments from the command line
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')

batch_size = 32
num_classes = 5
epochs = 2
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

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
    input_file_name = 'dataset_s.h5'
    external_input_file_path = os.path.join(FLAGS.input_dir, input_file_name);
    subprocess.check_call(['gsutil', '-m', 'cp', '-r',
                           external_input_file_path, '/tmp'])
    # Data loading and preprocessing, we copy it to the local machine inside the /tmp directory
    # Directly loading it from the bucket results in an error
    h5f = h5py.File(os.path.join('/tmp', input_file_name), 'r')
    X = h5f['X']
    Y = h5f['Y']
    X, Y = shuffle(X, Y)

    x_train, y_train, x_test, y_test = split_sets(X, Y, 0.2)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

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

    # Create the local checkpoint file and transfer it to the bucket afterwards
    internal_checkpoint_file = os.path.join('/tmp', 'checkpoint_' + model_name)
    external_checkpoint_file = os.path.join(FLAGS.output_dir, 'checkpoint_' + model_name)
    model.save(internal_checkpoint_file)
    # Use gsutil to copy the file from the tmp directory to the bucket
    subprocess.check_call(['gsutil', '-m', 'cp', '-r',
                           internal_checkpoint_file, external_checkpoint_file])
    print('Saved trained model at %s ' % external_checkpoint_file)


# Start the training for package
def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
