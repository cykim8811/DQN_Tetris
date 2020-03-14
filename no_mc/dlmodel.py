import tensorflow.keras as keras


def build_model():
    x_input = keras.Input(shape=(10, 20, 33))
    x = x_input

    x_shortcut = x
    x = keras.layers.Conv2D(33, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = x + x_shortcut

    x = keras.layers.Conv2D(12, (10, 1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(12, (1, 20), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(24)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(16)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(1)(x)

    x_output = x
    return keras.Model(x_input, x_output)
