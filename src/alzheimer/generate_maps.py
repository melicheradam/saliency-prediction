"""
Example usage of this script: python3 generate_generalized_maps.py -fix ./demo/fixations
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import shutil
import sys, os, inspect
import functools, operator
import multiprocessing as mp
import tqdm as tqdm
import math
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from helpers import fixpos_to_densemap, find_files_in_dir


INPUT_FIXATION_WIDTH = 1920
INPUT_FIXATION_HEIGHT = 1080
FINAL_MAP_WIDTH = 640
FINAL_MAP_HEIGHT = 360
NUM_OF_FIXATIONS_IN_MAP = 6
NUM_OF_PARALLEL_PROCESSES = 8
TAKE_LONGEST_FIXATIONS = True

# Some image file names do not match
def sanitize(name):
    replaced_chars = ['[', ']', '(', ')', ' ', '#', ',', "'"]
    for c in replaced_chars:
        name = name.replace(c, '')
    name = name.replace('..', '.')  # Some files contain name..jpg
    name = name.lower()
    return name


def generate_generalized_maps(fixations_dir, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    all_fixation_maps = find_files_in_dir(fixations_dir, filenameContains='.png') # Only saliency maps are jpg, Fixation maps are png
    unique_file_names = set([os.path.basename(f) for f in all_fixation_maps])

    for index, unique_file_name in enumerate(unique_file_names):

        matched_files = [f for f in all_fixation_maps if os.path.basename(f) == unique_file_name]
        generalized_map = np.zeros((FINAL_MAP_HEIGHT, FINAL_MAP_WIDTH), np.uint16)

        for matched_file in matched_files:
            matched_file_img = cv2.imread(matched_file, 0)

            if matched_file_img is None:
                print("Error loading image:", matched_file)
                continue

            if not matched_file_img.shape == generalized_map.shape:
                print("Warning: Resizing image from shape: ", matched_file_img.shape)
                matched_file_img = cv2.resize(matched_file_img, (generalized_map.shape[1], generalized_map.shape[0]))

            generalized_map = np.add(generalized_map, matched_file_img)  # Grayscale blurred image (0-255)

        # normalize map
        generalized_map = generalized_map / np.amax(generalized_map) * 255
        generalized_map = generalized_map.astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, unique_file_name), generalized_map)

        print(str(index) + "/" + str(len(unique_file_names)))


def produce_personalized_map_from_fixations(pack):
    image_record = pack[0]
    sm_output_dir = pack[1]
    fm_output_dir = pack[2]

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

    if TAKE_LONGEST_FIXATIONS:
        # Sort by fixation duration
        fixation_data = list(sorted(fixation_data, key=lambda r: r[2]))

    fixation_data = fixation_data[:NUM_OF_FIXATIONS_IN_MAP]
    fixation_data = np.asarray(fixation_data)

    # Create saliency map
    blank_image = np.zeros((INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, 3), np.uint8)
    # blank_image = blank_image.astype("uint8")
    saliency_map = fixpos_to_densemap(fixation_data, INPUT_FIXATION_WIDTH, INPUT_FIXATION_HEIGHT, blank_image)

    # Rescale the saliency map back to the size of the image (input fixation scale)
    saliency_map = cv2.resize(saliency_map, (FINAL_MAP_WIDTH, FINAL_MAP_HEIGHT), interpolation=cv2.INTER_AREA)
    saliency_map = saliency_map.astype("float32")
    saliency_map = saliency_map / np.amax(saliency_map) * 255.0
    saliency_map = saliency_map.astype("uint8")

    cv2.imwrite(os.path.join(sm_output_dir, image_name), saliency_map)

    # Create binary fixation map
    binary_map = np.zeros((FINAL_MAP_HEIGHT, FINAL_MAP_WIDTH, 3), np.uint8)
    # binary_map = binary_map.astype("uint8")

    for fixation in fixation_data:
        # Raw fixation positions are indexed from 1, not from 0
        x_pos = int(round(fixation[1] * ratio_x) - 1)
        y_pos = int(round(fixation[0] * ratio_y) - 1)
        # print(x_pos, y_pos)
        binary_map[x_pos, y_pos] = 255

    cv2.imwrite(os.path.join(fm_output_dir, os.path.splitext(image_name)[0] + ".png"), binary_map)


def generate_personalized_maps(fixations_dir, raw_fixations_dir):

    raw_fixation_files = find_files_in_dir(raw_fixations_dir, filenameContains='.json')
    subject_dirs = [f.path for f in os.scandir(os.path.join(fixations_dir, 'Personalized_maps')) if f.is_dir()]
    # subject_dirs = [f.path for f in os.scandir(fixations_dir) if f.is_dir()]

    # For each subject
    for subject_index, subdir in enumerate(subject_dirs):

        subject_name = os.path.basename(subdir)

        if 'Sub_' not in subject_name:
            continue

        fm_output_dir = os.path.join(subdir, 'FM')
        if os.path.exists(fm_output_dir):
            shutil.rmtree(fm_output_dir)
        os.mkdir(fm_output_dir)

        sm_output_dir = os.path.join(subdir, 'SM')
        if os.path.exists(sm_output_dir):
            shutil.rmtree(sm_output_dir)
        os.mkdir(sm_output_dir)

        # Find fixation file belonging to the subject
        belonging_fixation_file = [f for f in raw_fixation_files if subject_name in f][0]
        fixation_data = pd.read_json(belonging_fixation_file)

        pool = mp.Pool(NUM_OF_PARALLEL_PROCESSES)

        print("Generating personalized maps for subject: ", subject_name, " (", subject_index + 1, "/", len(subject_dirs), ")")
        results = list(
            tqdm.tqdm(
                pool.imap(
                    produce_personalized_map_from_fixations,
                    [(record, sm_output_dir, fm_output_dir) for record in fixation_data["ImfixDataALL"]]
                ),
                total=len(fixation_data["ImfixDataALL"])
            )
        )


def rename_sm_dir(fixations_dir):
    subject_dirs = [f.path for f in os.scandir(os.path.join(fixations_dir, 'Personalized_maps')) if f.is_dir()]

    # For each subject
    for index, subdir in enumerate(subject_dirs):
        if 'Sub_' not in subdir:
            continue

        output_dir = os.path.join(subdir, 'SM')
        if os.path.exists(output_dir):
            os.rename(output_dir, os.path.join(subdir, 'SM-original'))


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-fix", "--fixations-path", metavar="FP",
                         help="path for the fixations directory")

    #parser.add_argument("-raw", "--raw-fixations-path", metavar="FP",
    #                    help="path for the raw fixations directory")

    parser.add_argument("-out", "--output-path", metavar="OP",
                        help="path for the output generalized maps directory")

    args = parser.parse_args()
    # print("Generating personalized maps...")
    # generate_personalized_maps(args.fixations_path, args.raw_fixations_path)
    print("Generating generalized maps...")
    generate_generalized_maps(args.fixations_path, args.output_path)


if __name__ == "__main__":
    main()
