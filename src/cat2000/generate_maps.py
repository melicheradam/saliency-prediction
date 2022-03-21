import argparse
import functools
import inspect
import operator
import os
import sys
import tables
import pandas as pd
import numpy as np
import h5py
from pymatreader import read_mat


import cv2
import numpy as np
from scipy import io

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from helpers import find_files_in_dir
from psd.generate_maps import sanitize

INPUT_FIXATION_WIDTH = 1920
INPUT_FIXATION_HEIGHT = 1080
FINAL_MAP_WIDTH = 1920
FINAL_MAP_HEIGHT = 1080
NUM_OF_FIXATIONS_IN_MAP = 6


def produce_binary_map_from_fixations(file, output_dir):
    image_record = file
    print(file)
    image_name = image_record[0][0][0][0]
    image_name = sanitize(image_name)
    image_fixations_positions_y = functools.reduce(operator.iconcat, image_record[0][0][2])
    image_fixations_positions_x = functools.reduce(operator.iconcat, image_record[0][0][3])
    image_fixations_durations = functools.reduce(operator.iconcat, image_record[0][0][4])

    ratio_x = FINAL_MAP_WIDTH / INPUT_FIXATION_WIDTH
    ratio_y = FINAL_MAP_HEIGHT / INPUT_FIXATION_HEIGHT
    assert ratio_x == ratio_y

    fixation_data = [r for r in
                     zip(image_fixations_positions_x, image_fixations_positions_y, image_fixations_durations)]

    # fixation_data = fixation_data[:NUM_OF_FIXATIONS_IN_MAP]
    fixation_data = np.asarray(fixation_data)

    # Create binary fixation map
    binary_map = np.zeros((FINAL_MAP_HEIGHT, FINAL_MAP_WIDTH, 3), np.uint8)
    # binary_map = binary_map.astype("uint8")

    for fixation in fixation_data:
        # Raw fixation positions are indexed from 1, not from 0
        x_pos = int(round(fixation[1] * ratio_x) - 1)
        y_pos = int(round(fixation[0] * ratio_y) - 1)
        # print(x_pos, y_pos)
        binary_map[x_pos, y_pos] = 255

    cv2.imwrite(os.path.join(output_dir, os.path.splitext(image_name)[0] + ".png"), binary_map)


# Convert .mat fixation files to .csv
def convert_mat_to_json(fixations_dir):

    # mat_files = find_files_in_dir(os.path.join(fixations_dir, 'Raw'))
    mat_files = find_files_in_dir(os.path.join(fixations_dir), filenameContains='.mat')

    for index, mat_file in enumerate(mat_files):

        content = io.loadmat(mat_file)
        arr = np.array(content)
        print(content["fixLocs"])
        break
        mat = {k: v for k, v in content.items() if k[0] != '_'}
        print(mat)
        data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
        data.to_json(mat_file + '.json')
        print(index + 1, "/", len(mat_files))


def generate_binary_maps(raw_fixations_dir, output_dir):
    raw_fixation_files = find_files_in_dir(raw_fixations_dir, filenameContains='.mat')

    for file in raw_fixation_files:
        content = io.loadmat(file)
        arr = np.array(content["fixLocs"])
        # replace ones with 255
        binary_image = np.where(arr > 0, 255, arr)
        filename = str(file)[str(file).rfind("/") + 1:].replace(".mat", ".png")
        print(filename)
        cv2.imwrite(os.path.join(output_dir, filename), binary_image)


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-fix", "--fixations-path", metavar="FP",
                        help="path for the fixations directory")

    parser.add_argument("-out", "--output-path", metavar="FP",
                        help="path for the output directory")

    args = parser.parse_args()
    # convert_mat_to_json(args.fixations_path)
    generate_binary_maps(args.fixations_path, args.output_path)


if __name__ == "__main__":
    main()