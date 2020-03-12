import tensorflow.keras as keras


def build_model():
    x_input = keras.Input(shape=(10, 20, 33))
    x = x_input
    """
    x = keras.layers.Conv2D(16, (3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(16, (3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(16, (6, 1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    """
    x = keras.layers.Conv2D(3, (10, 1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(2, (1, 20), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    """
    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(32)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    """
    x = keras.layers.Dense(1)(x)

    x_output = x
    return keras.Model(x_input, x_output)
