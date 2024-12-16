import os, shutil
from os import listdir
from os.path import join
import shutil
import math
import random

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1


def create_split():
    """
    Split icon data into train/test/val sets based on the ratios defined above

    The function consolidate decks takes all the icons that were manually sorted by deck and puts them in a folder called icons
    This function keeps the images sorted by icons, and redistributes them across train/test/val datasets,
    and removes them from the icons folder to prevent repeats
    """
    if not os.path.exists(join("data", "train")):
        os.makedirs(join("data", "train"))
    if not os.path.exists(join("data", "test")):
        os.makedirs(join("data", "test"))
    if not os.path.exists(join("data", "validation")):
        os.makedirs(join("data", "validation"))
    icon_dirs = images = listdir(join("data", "icons"))

    for dir in icon_dirs:
        images = listdir(join("data", "icons", dir))
        num_train = math.ceil(len(images) * TRAIN_RATIO)
        num_test = math.floor(len(images) * TEST_RATIO)
        num_val = math.floor(len(images) * VAL_RATIO)
        # print(dir)
        # print(f"Train: {num_train}; Test: {num_test}; Val: {num_val}")
        test_images = random.sample(images, num_test)
        for image in test_images:
            # print(image)
            if not os.path.exists(join("data", "test", dir)):
                os.makedirs(join("data", "test", dir))
            shutil.copy(
                join("data", "icons", dir, image), join("data", "test", dir, image)
            )
            os.remove(join("data", "icons", dir, image))
        images = listdir(join("data", "icons", dir))
        val_images = random.sample(images, num_val)
        for image in val_images:
            # print(image)
            if not os.path.exists(join("data", "validation", dir)):
                os.makedirs(join("data", "validation", dir))
            shutil.copy(
                join("data", "icons", dir, image),
                join("data", "validation", dir, image),
            )
            os.remove(join("data", "icons", dir, image))
        train_images = listdir(join("data", "icons", dir))
        # train_images = random.sample(images, num_train)
        for image in train_images:
            if not os.path.exists(join("data", "train", dir)):
                os.makedirs(join("data", "train", dir))
            shutil.copy(
                join("data", "icons", dir, image), join("data", "train", dir, image)
            )
            os.remove(join("data", "icons", dir, image))
