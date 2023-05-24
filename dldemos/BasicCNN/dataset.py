import os
from typing import Tuple

import cv2
import numpy as np


def load_set(data_path: str, cnt: int, img_shape: Tuple[int, int]):
    cat_dirs = sorted(os.listdir(os.path.join(data_path, 'cats')))
    dog_dirs = sorted(os.listdir(os.path.join(data_path, 'dogs')))
    images = []
    for i, cat_dir in enumerate(cat_dirs):
        if i >= cnt:
            break
        name = os.path.join(data_path, 'cats', cat_dir)
        cat = cv2.imread(name)
        images.append(cat)

    for i, dog_dir in enumerate(dog_dirs):
        if i >= cnt:
            break
        name = os.path.join(data_path, 'dogs', dog_dir)
        dog = cv2.imread(name)
        images.append(dog)

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], img_shape)
        images[i] = images[i].astype(np.float32) / 255.0

    return np.array(images)


def get_cat_set(
        data_root: str,
        img_shape: Tuple[int, int] = (224, 224),
        train_size=1000,
        test_size=200,
        format='nhwc'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    train_X = load_set(os.path.join(data_root, 'training_set'), train_size,
                       img_shape)
    test_X = load_set(os.path.join(data_root, 'test_set'), test_size,
                      img_shape)

    train_Y = np.array([1] * train_size + [0] * train_size)
    test_Y = np.array([1] * test_size + [0] * test_size)

    if format == 'nhwc':
        return train_X, np.expand_dims(train_Y,
                                       1), test_X, np.expand_dims(test_Y, 1)
    elif format == 'nchw':
        train_X = np.reshape(train_X, (-1, 3, *img_shape))
        test_X = np.reshape(test_X, (-1, 3, *img_shape))
        return train_X, np.expand_dims(train_Y,
                                       1), test_X, np.expand_dims(test_Y, 1)
    else:
        raise NotImplementedError('Format must be "nhwc" or "nchw". ')
