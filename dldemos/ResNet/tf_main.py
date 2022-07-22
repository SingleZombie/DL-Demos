import tensorflow as tf
from tensorflow.keras import layers, models

from dldemos.BasicCNN.dataset import get_cat_set


def identity_block_2(x, f, use_shortcut=True):
    _, _, _, C = x.shape
    x_shortcut = x
    x = layers.Conv2D(C, f, padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(C, f, padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    if use_shortcut:
        x = x + x_shortcut
    x = layers.ReLU()(x)
    return x


def convolution_block_2(x, f, filters, s: int, use_shortcut=True):
    x_shortcut = x
    x = layers.Conv2D(filters, f, strides=(s, s), padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, f, padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    if use_shortcut:
        x_shortcut = layers.Conv2D(filters, 1, strides=(s, s),
                                   padding='valid')(x_shortcut)
        x_shortcut = layers.BatchNormalization(axis=3)(x_shortcut)
        x = x + x_shortcut
    x = layers.ReLU()(x)
    return x


def identity_block_3(x, f, filters1, filters2, use_shortcut=True):
    x_shortcut = x
    x = layers.Conv2D(filters1, 1, padding='valid')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Conv2D(filters1, f, padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters2, 1, padding='valid')(x)
    x = layers.BatchNormalization(axis=3)(x)
    if use_shortcut:
        x = x + x_shortcut
    x = layers.ReLU()(x)
    return x


def convolution_block_3(x, f, filters1, filters2, s: int, use_shortcut=True):
    x_shortcut = x
    x = layers.Conv2D(filters1, 1, strides=(s, s), padding='valid')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Conv2D(filters1, f, padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters2, 1, padding='valid')(x)
    x = layers.BatchNormalization(axis=3)(x)
    if use_shortcut:
        x_shortcut = layers.Conv2D(filters2,
                                   1,
                                   strides=(s, s),
                                   padding='valid')(x_shortcut)
        x_shortcut = layers.BatchNormalization(axis=3)(x_shortcut)
        x = x + x_shortcut
    x = layers.ReLU()(x)
    return x


def init_model(input_shape=(224, 224, 3),
               model_name='ResNet18',
               use_shortcut=True) -> tf.keras.models.Model:
    # Initialize input
    input = layers.Input(input_shape)

    # Get output
    x = layers.Conv2D(64, 7, (2, 2), padding='same')(input)
    x = layers.MaxPool2D((3, 3), (2, 2))(x)

    if model_name == 'ResNet18':
        x = identity_block_2(x, 3, use_shortcut)
        x = identity_block_2(x, 3, use_shortcut)
        x = convolution_block_2(x, 3, 128, 2, use_shortcut)
        x = identity_block_2(x, 3, use_shortcut)
        x = convolution_block_2(x, 3, 256, 2, use_shortcut)
        x = identity_block_2(x, 3, use_shortcut)
        x = convolution_block_2(x, 3, 512, 2, use_shortcut)
        x = identity_block_2(x, 3, use_shortcut)
    elif model_name == 'ResNet50':

        def block_group(x, fs1, fs2, count):
            x = convolution_block_3(x, 3, fs1, fs2, 2, use_shortcut)
            for i in range(count - 1):
                x = identity_block_3(x, 3, fs1, fs2, use_shortcut)
            return x

        x = block_group(x, 64, 256, 3)
        x = block_group(x, 128, 512, 4)
        x = block_group(x, 256, 1024, 6)
        x = block_group(x, 512, 2048, 3)
    else:
        raise NotImplementedError(f'No such model {model_name}')

    x = layers.AveragePooling2D((2, 2), (2, 2))(x)
    x = layers.Flatten()(x)
    output = layers.Dense(1, 'sigmoid')(x)

    # Build model
    model = models.Model(inputs=input, outputs=output)
    print(model.summary())
    return model


def main():
    train_X, train_Y, test_X, test_Y = get_cat_set(
        'dldemos/LogisticRegression/data/archive/dataset',
        train_size=500,
        test_size=50)
    print(train_X.shape)  # (m, 224, 224, 3)
    print(train_Y.shape)  # (m , 1)

    # model = init_model()
    # model = init_model(use_shortcut=False)
    model = init_model(model_name='ResNet50')
    # model = init_model(model_name='ResNet50', use_shortcut=False)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_X, train_Y, epochs=20, batch_size=16)
    model.evaluate(test_X, test_Y)


if __name__ == '__main__':
    main()
