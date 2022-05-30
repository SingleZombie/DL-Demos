import os
from typing import Tuple

import cv2
import numpy as np


def get_cat_set(
    data_root: str,
    img_shape: Tuple[int, int] = (224, 224),
    train_size=1000,
    test_size=200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def load_set(data_path: str, cnt: int):
        cat_dirs = os.listdir(os.path.join(data_path, 'cats'))
        dog_dirs = os.listdir(os.path.join(data_path, 'dogs'))
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
            images[i] = np.reshape(images[i], (-1))
            images[i] = images[i].astype(np.float32) / 255.0

        return np.array(images)

    train_X = load_set(os.path.join(data_root, 'training_set'), train_size)
    test_X = load_set(os.path.join(data_root, 'test_set'), test_size)

    train_Y = np.array([1] * train_size + [0] * train_size)
    test_Y = np.array([1] * test_size + [0] * test_size)

    return train_X.T, np.expand_dims(train_Y,
                                     0), test_X.T, np.expand_dims(test_Y.T, 0)
