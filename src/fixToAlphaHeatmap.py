"""
Example usage of this script:  python3 fixToAlphaHeatmap.py -fix eyetrackingdata_personalized/fixdens/Raw_Gauss_303/alex -orig eyetrackingdata_personalized/fixdens/Original\ Image\ Set
"""

import os
import argparse
from src.helpers import find_files_in_dir, parse_fixation_file, fixpos_to_densemap
import multiprocessing as mp
import cv2
import numpy as np

INPUT_FIXATION_WIDTH = 1024
INPUT_FIXATION_HEIGHT = 768

FINAL_MAP_WIDTH = 681
FINAL_MAP_HEIGHT = 511

# Create a saliency map for a given fixation file
def create_saliency_map(pack):

    fixation_file_path = pack[0]
    original_images_dir_path = pack[1]
    fixation_data = parse_fixation_file(fixation_file_path)
    original_images = sorted(find_files_in_dir(original_images_dir_path), key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    # Find corresponding original image based on the fixation file name
    index = int(os.path.basename(fixation_file_path).split('.')[0]) - 1
    corresponding_original_image_path = original_images[index]
    original_image = cv2.imread(corresponding_original_image_path)
    original_image = original_image.astype("uint8")

    head, filename = os.path.split(fixation_file_path)
    heatmap = fixpos_to_densemap(np.asarray(fixation_data), INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, original_image, color=cv2.COLORMAP_JET)
    output_path = os.path.join(os.path.dirname(fixation_file_path), 'SM-Alpha')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Rescale the saliency map back to the size of the image (input fixation scale)
    dim = (FINAL_MAP_WIDTH, FINAL_MAP_HEIGHT)
    heatmap_scaled = cv2.resize(heatmap, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path + '/' + filename.split('.')[0] + '.png', heatmap_scaled)
    return


def generate_saliency_maps(fixations_dir, original_images_dir):
    fixation_files = find_files_in_dir(fixations_dir)
    total_num_of_files = len(fixation_files)
    print("Total number of files to compute: ", total_num_of_files)
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(create_saliency_map, [(file, original_images_dir) for file in fixation_files])
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

    parser.add_argument("-orig", "--original-path", metavar="ORIG",
                        help="path for the original images directory")

    args = parser.parse_args()
    generate_saliency_maps(args.fixations_path, args.original_path)


if __name__ == "__main__":
    main()