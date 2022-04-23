"""
Example usage of this script: python3 differentiate_maps.py -gt ./gt -sal ./sal -out ./out -orig ./img
"""
import os
import argparse
import cv2
import datetime
import multiprocessing as mp
import tqdm
from helpers import fix_consistency
import numpy as np


def resize_images(gt, s_map, original_image):
    if not gt.shape == s_map.shape == original_image.shape:
        min_height = min(gt.shape[0], s_map.shape[0], original_image.shape[0])
        min_width = min(gt.shape[1], s_map.shape[1], original_image.shape[1])
        gt = cv2.resize(gt, (min_width, min_height))
        s_map = cv2.resize(s_map, (min_width, min_height))
        original_image = cv2.resize(original_image, (min_width, min_height))
    return gt, s_map, original_image


def produce_diff(pack):
    gt_path = pack[1][0]
    sal_path = pack[1][1]
    original_image_path = pack[1][2]
    out_path = pack[2]

    gt = cv2.imread(gt_path, 0)
    s_map = cv2.imread(sal_path, 0)
    original_image = cv2.imread(original_image_path, 0)  # Gray-scale background
    gt, s_map, original_image = resize_images(gt, s_map, original_image)

    # Convert maps to binary
    gt = np.where(gt < 10, False, gt)
    s_map = np.where(s_map < 10, False, s_map)
    gt = np.where(gt >= 10, True, gt)
    s_map = np.where(s_map >= 10, True, s_map)

    #  Mark TP green
    true_positives = np.logical_and(gt, s_map)
    tp_green = np.zeros((gt.shape[0], gt.shape[1], 3), np.uint16)
    tp_green[:, :, 0] = 0
    tp_green[:, :, 1] = true_positives * 255
    tp_green[:, :, 2] = 0

    # Mark FP Blue
    false_positives = s_map - gt
    fp_blue = np.zeros((gt.shape[0], gt.shape[1], 3), np.uint16)
    fp_blue[:, :, 0] = false_positives * 255
    fp_blue[:, :, 1] = 0
    fp_blue[:, :, 2] = 0

    # Mark FN Red
    false_negatives = gt - s_map
    fn_red = np.zeros((gt.shape[0], gt.shape[1], 3), np.uint16)
    fn_red[:, :, 0] = 0
    fn_red[:, :, 1] = 0
    fn_red[:, :, 2] = false_negatives * 255

    # Add them up
    tp_green = tp_green.astype(np.uint8)
    fp_blue = fp_blue.astype(np.uint8)
    fn_red = fn_red.astype(np.uint8)

    diff = cv2.addWeighted(tp_green, 1.0, fp_blue, 1.0, 0.0)
    diff = cv2.addWeighted(diff, 1.0, fn_red, 1.0, 0.0)
    diff = cv2.addWeighted(cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB), 0.5, diff, 1.0, 0.0)

    # Add color legend
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, diff.shape[1] - 180)
    font_scale = 0.8
    line_type = 2

    font_color = (0, 0, 255)
    position = bottom_left_corner
    cv2.putText(diff, 'Red = False positives',
                position,
                font,
                font_scale,
                font_color,
                line_type)

    font_color = (0, 255, 0)
    position = (position[0], position[1] - 30)
    cv2.putText(diff, 'Green = True positives',
                position,
                font,
                font_scale,
                font_color,
                line_type)

    font_color = (255, 0, 0)
    position = (position[0], position[1] - 30)
    cv2.putText(diff, 'Blue = False negatives',
                position,
                font,
                font_scale,
                font_color,
                line_type)
    """
    cv2.imwrite(out_path + '/' + os.path.splitext(os.path.basename(gt_path))[0] + '.png', diff)


def traverse_dirs(gt_directory, sm_directory, output_path, original_images_path):

    start = datetime.datetime.now().replace(microsecond=0)

    # Find paths of all grountruth and predicted maps and sort them
    gt_dir_path = os.path.join(gt_directory)
    sm_dir_path = os.path.join(sm_directory)
    orig_images_dir_path = os.path.join(original_images_path)
    output_dir_path = os.path.join(output_path)

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    gt_files = os.listdir(gt_dir_path)
    sm_files = os.listdir(sm_dir_path)
    orig_image_files = os.listdir(orig_images_dir_path)

    gt_files, sm_files = fix_consistency(gt_files, sm_files)
    gt_files, orig_image_files = fix_consistency(gt_files, orig_image_files)
    orig_image_files, sm_files = fix_consistency(orig_image_files, sm_files)

    gt_files_sorted = sorted(gt_files, key=lambda i: os.path.splitext(os.path.basename(i))[0])
    sm_files_sorted = sorted(sm_files, key=lambda i: os.path.splitext(os.path.basename(i))[0])
    orig_image_files_sorted = sorted(orig_image_files, key=lambda i: os.path.splitext(os.path.basename(i))[0])

    gt_paths = []
    for file in gt_files_sorted:
        gt_paths.append(os.path.join(gt_directory, file))
    sm_paths = []
    for file in sm_files_sorted:
        sm_paths.append(os.path.join(sm_directory, file))
    orig_paths = []
    for file in orig_image_files_sorted:
        orig_paths.append(os.path.join(original_images_path, file))

    assert len(gt_paths) == len(sm_paths) == len(orig_paths)
    total_num_of_files = len(gt_files_sorted)
    print("Total number of files to compare: ", total_num_of_files)

    pool = mp.Pool(mp.cpu_count() // 2)
    results = list(tqdm.tqdm(
        pool.imap(produce_diff, [(i, (a, b, c), output_dir_path) for (i, (a, b, c)) in enumerate(zip(gt_paths, sm_paths, orig_paths))]),
        total=total_num_of_files))


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    import time
    start = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-gt", "--ground-truth", metavar="GT",
                        help="path for groundtruth saliency maps directory")

    parser.add_argument("-sal", "--saliency-maps", metavar="SAL",
                        help="path for result saliency maps directory")

    parser.add_argument("-orig", "--original-images", metavar="ORI",
                        help="path for the directory containing original images")

    parser.add_argument("-output", "--output-path", metavar="OUT",
                        help="file path where results will be stored")

    args = parser.parse_args()
    traverse_dirs(args.ground_truth, args.saliency_maps, args.output_path, args.original_images)

    end = time.time()

    print('Total elapsed time:', end - start)


if __name__ == "__main__":
    main()