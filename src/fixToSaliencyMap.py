"""
Example usage of this script: python3 fixToSaliencyMap.py -fix ./demo/fixations
"""

import os
import argparse
from helpers import find_files_in_dir, parse_fixation_file, fixpos_to_densemap
import multiprocessing as mp
import cv2
import numpy as np

INPUT_FIXATION_WIDTH = 1024
INPUT_FIXATION_HEIGHT = 768

FINAL_MAP_WIDTH = 681
FINAL_MAP_HEIGHT = 511

MERGE_CLOSE_FIXATIONS = False
PARALLEL_PROCESSES = 3

# Create a saliency map for a given fixation file
def create_saliency_map(fixation_file_path):

    fixation_data = parse_fixation_file(fixation_file_path, merge_fixations=MERGE_CLOSE_FIXATIONS)
    if len(fixation_data) == 0:
        return

    # Produce saliency map
    fixation_data = np.asarray(fixation_data)
    blank_image = np.zeros((INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, 3), np.uint8)
    blank_image = blank_image.astype("uint8")
    heatmap = fixpos_to_densemap(fixation_data, INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, blank_image)

    # Rescale the saliency map back to the size of the image (input fixation scale)
    dim = (FINAL_MAP_WIDTH, FINAL_MAP_HEIGHT)
    heatmap_scaled = cv2.resize(heatmap, dim, interpolation=cv2.INTER_AREA)

    heatmap_scaled = heatmap_scaled.astype("float32")
    heatmap_scaled = heatmap_scaled / np.amax(heatmap_scaled)
    heatmap_scaled = heatmap_scaled * 255.0
    heatmap_scaled = heatmap_scaled.astype("uint8")

    # Store the saliency map
    head, filename = os.path.split(fixation_file_path)
    output_path = os.path.join(os.path.dirname(fixation_file_path), 'SM')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(output_path + '/' + filename.split('.')[0] + '.png', heatmap_scaled)

    # Also store the original binary fixation map
    binary_fixation_map = np.zeros((INPUT_FIXATION_HEIGHT, INPUT_FIXATION_WIDTH), np.uint8)
    binary_fixation_map = binary_fixation_map.astype("uint8")

    for fixation in fixation_data:
        binary_fixation_map[int(fixation[1]), int(fixation[0])] = 255

    # Rescale the saliency map back to the size of the image (input fixation scale)
    binary_fixation_map_scaled = cv2.resize(binary_fixation_map, dim)
    binary_fixation_map_scaled[binary_fixation_map_scaled > 0] = 255

    output_path = os.path.join(os.path.dirname(fixation_file_path), 'FM')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(output_path + '/' + filename.split('.')[0] + '.png', binary_fixation_map_scaled)

    return

def generate_saliency_maps(fixations_dir):

    fixation_files = find_files_in_dir(fixations_dir, filenameContains='.fix')
    total_num_of_files = len(fixation_files)
    print("Total number of files to compute: ", total_num_of_files)
    pool = mp.Pool(PARALLEL_PROCESSES)
    results = pool.map(create_saliency_map, [file for file in fixation_files])

    return


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-fix", "--fixations-path", metavar="FP",
                        help="path for the fixations directory")

    args = parser.parse_args()
    generate_saliency_maps(args.fixations_path)


if __name__ == "__main__":
    main()