
# Implementation of CNN Classifier

import CNN_classifier
import keras
from keras.datasets import cifar10
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger
import numpy as np

# Setting learning rate reducer, Default : patience = 3, cooldown = 2
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, cooldown=3, min_lr=0)
# Save logfile
csv_logger = CSVLogger('CNN_cifar10.csv')

# Hyperparameter
batch_size = 32             # Default = 64
input_shape = (32, 32, 3)
nb_classes = 10
nb_epoch = 100
kernel_regularizer = 1.e-4  # Default = 1.e-2, but too large
lr = 0.001               # Default = 0.001
op = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
data_augmentation = True

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Reshape labels
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
# Data preprocessing
mean_image = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
# Normalization
x_train -= mean_image
x_test -= mean_image
x_train /= (std+1e-7)       # Sum very small number to prevent 0
x_test /= (std+1e-7)

# Create model
model = CNN_classifier.vgg_c(input_shape, nb_classes, kernel_regularizer, weights_path=None)
# Configures the model for training
model.compile(loss='categorical_crossentropy',
              optimizer=op,
              metrics=['accuracy'])

## Training
# No data_augmentation
if not data_augmentation:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(x_test, y_test),
              shuffle=True,                         # Shuffles in batch-sized chunks
              callbacks=[reduce_lr, csv_logger])     # Apply reduce_lr, csv_logger
# Using data_augmentation
else:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,                       # Default = 0
        width_shift_range=0.1,                  # Default = 0.1
        height_shift_range=0.1,                 # Default = 0.1
        horizontal_flip=True,
        vertical_flip=False)
    # Data augmentation to x_train
    datagen.fit(x_train)
    # Fits the model on data generated batch by batch
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,         # Total number of steps
                        validation_data=(x_test, y_test),
                        epochs=nb_epoch,
                        verbose=1,                                              # Show progress bar
                        max_queue_size=100,                                     # Maximum size for the generator
                        callbacks=[reduce_lr, csv_logger])
# Save weights
model.save_weights('CNN_cifar10.h5')
