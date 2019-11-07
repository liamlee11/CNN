# CNN classifier
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten
)
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    BatchNormalization
)
from keras import regularizers

## vgg_a model
def vgg_a(input_shape, num_outputs, rs, weights_path=None):
    # Make sequential model
    model = Sequential()
    # Block 1
    model.add(
        Conv2D(64, (3, 3),                                  # 64 filters, 3 x 3 size
               padding='same',                             # Padding
               input_shape=input_shape,                     # [32, 32, 3]
               kernel_regularizer=regularizers.l2(rs)))     # L2 reqularization
    model.add(BatchNormalization())                         # Batch normalization
    model.add(Activation('relu'))                          # Activation function
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # MaxPooling layer, 2 x 2 size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 5
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FC layers
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(rs)))
    model.add(Activation('relu'))
    model.add(Dense(512, kernel_regularizer=regularizers.l2(rs)))
    model.add(Activation('relu'))
    # Softmax
    model.add(Dense(num_outputs, activation='softmax'))
    # Load weights if not None
    if weights_path:
        model.load_weights(weights_path)

    return model
## vgg_b model
def vgg_b(input_shape, num_outputs, rs, weights_path=None):
    model = Sequential()
    # Block 1
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               input_shape=input_shape,
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 5
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FC layer and softmax
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
## vgg_c model
def vgg_c(input_shape, num_outputs, rs, weights_path=None):
    model = Sequential()
    # Block 1
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               input_shape=input_shape,
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))                                 # Dropout layer
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 5
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FC layer and softmax
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
## vgg_d model
def vgg_d(input_shape, num_outputs, rs, weights_path=None):
    model = Sequential()
    # Block 1
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               input_shape=input_shape,
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(64, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(128, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(256, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 5
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
        Conv2D(512, (3, 3),
               padding='same',
               kernel_regularizer=regularizers.l2(rs)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FC layer and softmax
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model