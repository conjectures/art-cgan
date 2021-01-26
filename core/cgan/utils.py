
import os
import glob
import numpy as np
import math
from PIL import Image


# Check if 
def mkdir_if_not_exist(directory):
    if not (os.path.isdir(directory)):
        os.mkdir(dirname)


def check_filename(filename, dir, ext=None):
    """ Check if filename exists and change suffix accordingly """
    # Count number of files that contain the value of filename
    counter = len([file for file in os.listdir(dir) if filename in file])
    suffix = '' if ext is None else f".{ext}"
    path = os.path.join(dir, f"{filename}_{counter:02d}{suffix}")
    print(path)
    return path



def load_image(filename):
    " Load image with Pillow into numpy array"
    img = Image.open(filename).convert("RGB")
    img = np.asarray(img, dtype="int8")
    return img


def get_images(data_dir, image_size=512, channels=3):
    """ Load images from path and create training data images and labels"""
    # iterate dirs
    dirs = [directory for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory))]
    num_classes = len(dirs)
    print(f"{num_classes=}")

    image_list = []
    label_list = []

    # Get paths and labels of the training images
    for label, dir in enumerate(dirs):
        for itm in os.listdir(os.path.join(data_dir, dir)):
           img_path = os.path.join(data_dir, dir, itm) 
           image_list.append(img_path)
           label_list.append(label)

    # Get total number of images to create numpy array
    num_images = len(label_list)
    x_train = np.zeros((num_images, image_size, image_size, channels), dtype='int8')
    label_list = np.asarray(label_list, dtype='int8')
    # shallow copy
    y_train = label_list

    # Load images and store into numpy list
    for idx, itm in enumerate(image_list):
        img = load_image(itm)
        x_train[idx, :, :, :] = img

    return {
            'train_data': (x_train, y_train),
            'num_classes': num_classes,
            }


def get_training_data(data_dir):
    """ Process data for training """
    # TODO: shuffle x_train and y_train (together)
    return get_images(data_dir)

