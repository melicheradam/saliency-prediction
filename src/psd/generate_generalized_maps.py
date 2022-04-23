"""
Example usage of this script: python3 generate_generalized_maps.py -fix ./demo/fixations
"""
import os
import argparse
import cv2
import numpy as np


INPUT_FIXATION_WIDTH = 1920
INPUT_FIXATION_HEIGHT = 1080
FINAL_MAP_WIDTH = INPUT_FIXATION_WIDTH
FINAL_MAP_HEIGHT = INPUT_FIXATION_HEIGHT

# Return files inside the provided directory and subdirectories
# TODO fix package imports and replace with the function from helpers.py
def find_files_in_dir(dirpath, filename=None, filenameContains=None):
    files = []
    for root, dirs, localfiles in os.walk(dirpath):
        for f in localfiles:
            # If filename is provided, filter only files matching the name
            if filename:
                if os.path.basename(f) == filename:
                    files.append(os.path.join(root, f))
            # If file name contains is provided
            elif filenameContains:
                if filenameContains in os.path.basename(f):
                    files.append(os.path.join(root, f))
            else:
                files.append(os.path.join(root, f))
    return files


def generate_generalized_maps(fixations_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_fixation_maps = find_files_in_dir(fixations_dir)
    unique_file_names = set([os.path.basename(f) for f in all_fixation_maps])

    for index, unique_file_name in enumerate(unique_file_names):

        matched_files = [f for f in all_fixation_maps if os.path.basename(f) == unique_file_name]
        generalized_map = np.zeros((FINAL_MAP_HEIGHT, FINAL_MAP_WIDTH), np.uint16)

        for matched_file in matched_files:
            matched_file_img = cv2.imread(matched_file, 0)

            if not matched_file_img.shape == generalized_map.shape:
                print("Warning: Resizing image from shape: ", matched_file_img.shape)
                matched_file_img = cv2.resize(matched_file_img, (generalized_map.shape[1], generalized_map.shape[0]))

            generalized_map = np.add(generalized_map, matched_file_img)  # Grayscale blurred image (0-255)

        # normalize map
        generalized_map = generalized_map / np.amax(generalized_map) * 255
        cv2.imwrite(os.path.join(output_dir, unique_file_name), generalized_map)

        print(str(index) + "/" + str(len(unique_file_names)))


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
    generate_generalized_maps(args.fixations_path, args.output_path)


if __name__ == "__main__":
    main()
