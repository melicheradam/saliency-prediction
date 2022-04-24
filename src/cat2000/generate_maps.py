import argparse
import functools
import operator
import os
import pandas as pd
import cv2
import numpy as np
from scipy import io
from src.helpers import find_files_in_dir
from src.psd.generate_maps import sanitize

INPUT_FIXATION_WIDTH = 1920
INPUT_FIXATION_HEIGHT = 1080
FINAL_MAP_WIDTH = 1920
FINAL_MAP_HEIGHT = 1080
NUM_OF_FIXATIONS_IN_MAP = 6


def generate_binary_maps(raw_fixations_dir, output_dir):
    raw_fixation_files = find_files_in_dir(raw_fixations_dir, filenameContains='.mat')

    for file in raw_fixation_files:
        content = io.loadmat(file)
        arr = np.array(content["fixLocs"])
        # replace ones with 255
        binary_image = np.where(arr > 0, 255, arr)
        filename = str(file)[str(file).rfind("/") + 1:].replace(".mat", ".png")
        cv2.imwrite(os.path.join(output_dir, filename), binary_image)


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-raw", "--raw-fixations-path", metavar="RW",
                        help="path for the raw fixations directory (.mat)")

    parser.add_argument("-out", "--output-path", metavar="FP",
                        help="path for the output directory")

    args = parser.parse_args()

    generate_binary_maps(args.raw_fixations_path, args.output_path)


if __name__ == "__main__":
    main()
