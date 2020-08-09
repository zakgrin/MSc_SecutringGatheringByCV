import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def init_weights(shape, dtype=None, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def init_bias(shape, dtype=None, name=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_conv(input_shape, name=None, model=None, api='sequential'):
    if api == 'functional':
        if model == None:
            input_layer = keras.Input(input_shape)
            x_1 = layers.Conv2D(64, (10, 10), activation='relu', kernel_regularizer=l2(2e-4), input_shape=input_shape,
                                kernel_initializer=init_weights, bias_initializer=init_bias)(input_layer)
            x_2 = layers.MaxPooling2D(strides=2)(x_1)
            x_3 = layers.Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4),
                                kernel_initializer=init_weights, bias_initializer=init_bias)(x_2)
            x_4 = layers.MaxPooling2D(strides=2)(x_3)
            x_5 = layers.Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4),
                                kernel_initializer=init_weights, bias_initializer=init_bias)(x_4)
            x_6 = layers.MaxPooling2D(strides=2)(x_5)
            x_7 = layers.Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4),
                                kernel_initializer=init_weights, bias_initializer=init_bias)(x_6)
            x_8 = layers.Flatten()(x_7)
            x_9 = layers.Dense(4096, activation='sigmoid', kernel_regularizer=l2(2e-4),
                               kernel_initializer=init_weights, bias_initializer=init_bias)(x_8)
            encoded = keras.Model(inputs=input_layer, outputs=x_9, name=name)
            return encoded, input_layer
    elif api == 'sequential':
        input_layer = keras.Input(input_shape, name='{}_input_layer'.format(name))
        if model == None:
            model = keras.Sequential(name='sequential')
            model.add(
                layers.Conv2D(64, (10, 10), activation='relu', kernel_regularizer=l2(2e-4), input_shape=input_shape,
                              kernel_initializer=init_weights, bias_initializer=init_bias))
            model.add(layers.MaxPooling2D(strides=2))
            model.add(layers.Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4),
                                    kernel_initializer=init_weights, bias_initializer=init_bias))
            model.add(layers.MaxPooling2D(strides=2))
            model.add(layers.Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4),
                                    kernel_initializer=init_weights, bias_initializer=init_bias))
            model.add(layers.MaxPooling2D(strides=2))
            model.add(layers.Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4),
                                    kernel_initializer=init_weights, bias_initializer=init_bias))
            model.add(layers.Flatten())
            model.add(layers.Dense(4096, activation='sigmoid', kernel_regularizer=l2(2e-4),
                                   kernel_initializer=init_weights, bias_initializer=init_bias))
            encoded = model(input_layer)
            return encoded, input_layer, model
        else:
            encoded = model(input_layer)
            return encoded, input_layer
    elif api == 'embeds':
        input_layer = keras.Input(input_shape, name='{}_input_layer'.format(name))
        if model == None:
            model = keras.Sequential(name='sequential')
            model.add(layers.Flatten())
            # model.add(layers.Dense(64, activation='relu'))
            # model.add(layers.Dense(512, activation='relu'))
            # model.add(layers.Dense(512, activation='sigmoid'))
            # model.add(layers.Dense(512, activation='sigmoid'))
            encoded = model(input_layer)
            return encoded, input_layer, model
        else:
            encoded = model(input_layer)
            return encoded, input_layer


def init_siamse_model(input_shape, api='sequential'):
    # encode
    left_encoded, left_input, model = get_conv(input_shape, name="left", api=api)
    right_encoded, right_input = get_conv(input_shape, name="right", model=model, api=api)
    # compute the absolute difference (cosine loss function)
    L1_layer = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='abs_diff')
    L1_distance = L1_layer([left_encoded, right_encoded])
    # similarity score
    y = layers.Dense(1, activation='sigmoid', bias_initializer=init_bias, name='ouput')(L1_distance)
    # siamese network
    siamese_net = keras.Model(inputs=[left_input, right_input], outputs=y)
    return siamese_net