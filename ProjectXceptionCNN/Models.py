import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, LeakyReLU
from tensorflow.keras.layers import Input, Dense, Conv2D, Add
from tensorflow.keras.layers import SeparableConv2D, ReLU
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3


# Xception Model architecture

def get_xception(input_size, loss='categorical_crossentropy', metrics=['accuracy'], classes_num=102,
                 optimizer=RMSprop(learning_rate=0.001)):
    tf.keras.backend.clear_session()
    model = Xception(include_top=False, input_shape=input_size, weights='imagenet')
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, activation='relu')(flat1)
    output = Dense(classes_num, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# InceptionV3 Model architecture

def get_inceptionv3(input_size, loss='categorical_crossentropy', metrics=['accuracy'], classes_num=102,
                    optimizer=RMSprop(learning_rate=0.001)):
    tf.keras.backend.clear_session()
    model = InceptionV3(include_top=False, input_shape=input_size, weights='imagenet')
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, activation='relu')(flat1)
    output = Dense(classes_num, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# Xception Model architecture with dropout layers

def get_Xception_dropout(input_size, loss='categorical_crossentropy', metrics=['accuracy'], classes_num=102,
                         optimizer=RMSprop(learning_rate=0.001)):
    tf.keras.backend.clear_session()
    input = Input(shape=input_size)
    x = entry_flow(input)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = middle_flow(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = exit_flow(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    flat1 = Flatten()(x)
    class1 = Dense(256, activation='relu')(flat1)
    output = Dense(classes_num, activation='softmax')(class1)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# creating the Conv-Batch Norm block

def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


# creating separableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


# entry flow

def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    x = Add()([tensor, x])
    return x


# middle flow

def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        tensor = Add()([tensor, x])

    return tensor


# exit flow

def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = GlobalAvgPool2D()(x)

    x = Dense(units=1000, activation='softmax')(x)

    return x
