"""
Example usage of this script: python3 fixToGeneralizedMap.py -fix ./demo/fixations
"""

import os
import argparse
from helpers import find_files_in_dir, parse_fixation_file, fixpos_to_densemap, delete_dir_with_content
import multiprocessing as mp
import cv2
import numpy as np

INPUT_FIXATION_WIDTH = 1024
INPUT_FIXATION_HEIGHT = 768

FINAL_MAP_WIDTH = 681
FINAL_MAP_HEIGHT = 511

MERGE_CLOSE_FIXATIONS = False


# Create a saliency map for a given fixation file
def create_saliency_map(pack):
    file_name = pack[0]
    output_dir_path = pack[1]
    fixations_dir_path = pack[2]

    # Group same .fix files across all observers
    matched_files = find_files_in_dir(fixations_dir_path, filename=file_name)

    # Produce a generalized map based on these .fix files
    fixations = []
    for file in matched_files:
        fixations.append(parse_fixation_file(file, merge_fixations=MERGE_CLOSE_FIXATIONS))
    fixations = [item for sublist in fixations for item in sublist]  # Flatten the list

    merged_fixations = fixations

    blank_image = np.zeros((INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, 3), np.uint8)
    blank_image = blank_image.astype("uint8")

    heatmap = fixpos_to_densemap(np.asarray(merged_fixations), INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, blank_image)

    # Rescale the saliency map back to the size of the image (input fixation scale)
    dim = (FINAL_MAP_WIDTH, FINAL_MAP_HEIGHT)
    heatmap_scaled = cv2.resize(heatmap, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_dir_path + '/' + file_name.split('.')[0] + '.png', heatmap_scaled)
    return


def generate_saliency_maps(fixations_dir, output_dir):
    fixation_files = find_files_in_dir(fixations_dir)
    total_num_of_files = len(fixation_files)
    print("Total number of files to compute: ", total_num_of_files)

    if os.path.exists(output_dir):
        delete_dir_with_content(output_dir)
    os.makedirs(output_dir)

    # Find unique .fix file names and sort them
    unique_fixation_file_names = set()
    for fixation_file in fixation_files:
        file_name = os.path.basename(fixation_file)
        if len(file_name.split('.')) == 2 and file_name.split('.')[1] == 'fix':
            unique_fixation_file_names.add(file_name)
    unique_fixation_file_names = sorted(unique_fixation_file_names, key=lambda i: int(i.split('.')[0]))

    pool = mp.Pool(mp.cpu_count() // 2)
    results = pool.map(create_saliency_map, [(file, output_dir, fixations_dir) for file in unique_fixation_file_names])
    return


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-fix", "--fixations-path", metavar="FP",
                        help="path for the fixations directory")

    parser.add_argument("-out", "--output-path", metavar="OP",
                        help="path for the output generalized maps directory")

    args = parser.parse_args()
    generate_saliency_maps(args.fixations_path, args.output_path)


if __name__ == "__main__":
    main()
