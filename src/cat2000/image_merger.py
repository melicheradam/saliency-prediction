import argparse
import ntpath
import os
from pathlib import Path

import cv2
import numpy as np
from src.helpers import find_files_in_dir

IMG_H = 1080
IMG_W = 1920
SUB_NAMES = [None]


def get_sub_names(model_path):
    items = os.listdir(os.path.join(model_path))
    filtered = [item for item in items if os.path.isdir(os.path.join(model_path, item))
                and (str(item) != "saliency" or str(item) != "discrepancy")]
    return filtered


def read_images(image_name, model_path):
    images = []
    global SUB_NAMES
    for sub in SUB_NAMES:
        final_path = model_path.joinpath(str(sub)).joinpath("saliency").joinpath(image_name)
        images.append(cv2.imread(str(final_path), cv2.IMREAD_COLOR))

    return images


def combine_all(res_path: Path, model_name: str):
    # unique file names
    model_path = res_path.joinpath(model_name)
    global SUB_NAMES
    SUB_NAMES = get_sub_names(model_path)
    img_names = set(find_files_in_dir(model_path, filenameContains=".jpeg"))

    for img_name in img_names:
        img_name = ntpath.basename(img_name)
        images = read_images(img_name, model_path)

        # create blank black image
        dest = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        for image in images:
            # added images have proportional weight no number of subjects
            dest = cv2.addWeighted(dest, 1, image, 1 / len(SUB_NAMES), 0.0)

        # Saving the output image
        cv2.imwrite(os.path.join(res_path, model_name + "_merged", img_name), dest)


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-name", "--model-name", metavar="MN",
                        help="name of the model which was evaluated")

    parser.add_argument("-res", "--results-dir", metavar="RD",
                        help="path to results directory")

    args = parser.parse_args()
    res_path = Path(args.results_dir)
    Path(res_path).joinpath(args.model_name + "_merged").mkdir(parents=True, exist_ok=True)
    print("Combining saliency maps...")
    combine_all(res_path, args.model_name)


if __name__ == "__main__":
    main()




