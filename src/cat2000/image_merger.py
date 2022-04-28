import os
from pathlib import Path

import cv2
import numpy as np
from ..helpers import find_files_in_dir
from config import RESULTS_DIR

SUB_NAMES = []
IMG_H = 2000
IMG_W = 2000


def get_sub_names(model_name):
    items = os.listdir(os.path.join(model_name))
    filtered = [item for item in items if os.path.isdir(os.path.join(model_name, item))]
    return filtered


def read_images(image_name, model_name):
    images = []
    for sub in get_sub_names(model_name):
        sub_dir = os.path.join(RESULTS_DIR, model_name, sub)
        images.append(cv2.imread(os.path.join(sub_dir, "saliency", image_name), cv2.IMREAD_COLOR))

    return images


def combine_all(model_name):
    # unique file names
    img_names = set(find_files_in_dir("", filenameContains=".jpeg"))

    for img_name in img_names:
        images = read_images(img_name, model_name)

        # create blank black image
        dest = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        for image in images:
            # added images have proportional weight no number of subjects
            dest = cv2.addWeighted(dest, 1, image, 1 / len(SUB_NAMES), 0.0)

        # Saving the output image
        cv2.imwrite(os.path.join(RESULTS_DIR, model_name + "_merged", img_name), dest)


if __name__ == '__main__':
    combine_all('PyCharm')

