import argparse
import os
import cv2
import numpy as np
from helpers import find_files_in_dir, load_sub_names
from scipy.ndimage import gaussian_filter

SUB_NAMES = []

def read_images(dirname, image_name):
    images = []

    for sub in SUB_NAMES:
        images.append(cv2.imread(os.path.join(*[dirname, sub, "cat2000-saliency", image_name]), cv2.IMREAD_COLOR))

    return images


def combine_all(dirname, g_value, thresh_value):
    # unique file names
    img_names = find_files_in_dir(dirname, filenameContains=".jpeg")
    img_names = [os.path.basename(item) for item in img_names]
    img_names = set(img_names)

    out_dirname = "merged_images" + str(thresh_value) + "_" + str(g_value)
    if not os.path.exists(os.path.join(*[dirname, out_dirname])):
        os.mkdir(os.path.join(*[dirname, out_dirname]))

    for img_name in img_names:
        images = read_images(dirname, img_name)

        for idx, image in enumerate(images):
            if idx == 0:
                dest = image
            else:
                # added images have proportional weight no number of subjects
                dest = cv2.addWeighted(dest, 1 - (1 / len(SUB_NAMES)), image, 1 / len(SUB_NAMES), 0.0)

        # Saving the output image
        #_, dest = cv2.threshold(dest, int(thresh_value), 255, cv2.THRESH_TOZERO)
        #dest = gaussian_filter(dest, sigma=int(g_value))


        cv2.imwrite(os.path.join(*[dirname, out_dirname, img_name]), dest)


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-sal", "--saliency-maps", metavar="FP",
                        help="path for the saliency maps directory")

    parser.add_argument("-sub", "--subject-names", metavar="FP",
                        help="path for the subject names file")

    parser.add_argument("-g", "--gfilter-value", metavar="FP",
                        help="path for the subject names file")

    parser.add_argument("-th", "--thresh-value", metavar="FP",
                        help="path for the subject names file")

    args = parser.parse_args()

    global SUB_NAMES
    SUB_NAMES = load_sub_names(args.subject_names)

    combine_all(args.saliency_maps, args.gfilter_value, args.thresh_value)


if __name__ == "__main__":
    main()

