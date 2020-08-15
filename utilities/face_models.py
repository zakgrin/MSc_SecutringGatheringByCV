import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def init_weights(shape, dtype=None, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def init_bias(shape, dtype=None, name=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_conv(input_shape, name=None, model=None, app='images', learn='before_l2'):
    if app == 'images':
        input_layer = keras.Input(input_shape, name='{}_input_layer'.format(name))
        if model == None:
            model = keras.Sequential(name='siamese_{}'.format(app))
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
    elif app == 'embeds':
        input_layer = keras.Input(input_shape, name='{}_input_layer'.format(name))
        if model == None:
            model = keras.Sequential(name='siamese_{}'.format(app))
            model.add(layers.Flatten())
            if learn == 'before_l2' or learn == 'before_after_l2':
                model.add(layers.Dense(256, activation='relu'))
            encoded = model(input_layer)
            return encoded, input_layer, model
        else:
            encoded = model(input_layer)
            return encoded, input_layer


def init_siamse_model(input_shape, app='images', learn='before_l2'):
    if app == 'images' and (learn == 'before_l2' or learn == 'only_l2'):
        print("When app='images', learn='before_l2 and learn='only_l2' give exactly the same model")
        print(" - in both cases, a learning layers has to be added before calculating l2 distance")

    # encode
    left_encoded, left_input, model = get_conv(input_shape, name="left", app=app, learn=learn)
    right_encoded, right_input = get_conv(input_shape, name="right", model=model, app=app)

    # compute the absolute difference (cosine loss function)
    L1_layer = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='abs_diff')
    L1_distance = L1_layer([left_encoded, right_encoded])

    if learn == 'after_l2' or learn == 'before_after_l2':
        learn_layer = layers.Dense(64, activation='sigmoid', kernel_regularizer=l2(2e-4),
                                   kernel_initializer=init_weights, bias_initializer=init_bias)(L1_distance)
        learn_layer = layers.Dense(32, activation='sigmoid', kernel_regularizer=l2(2e-4),
                                   kernel_initializer=init_weights, bias_initializer=init_bias)(learn_layer)
        y = layers.Dense(1, activation='sigmoid', bias_initializer=init_bias, name='ouput')(learn_layer)

    elif learn == 'only_l2' or learn == 'before_l2':
        y = layers.Dense(1, activation='sigmoid', bias_initializer=init_bias, name='ouput')(L1_distance)

    # siamese network
    siamese_net = keras.Model(inputs=[left_input, right_input], outputs=y)
    return siamese_net