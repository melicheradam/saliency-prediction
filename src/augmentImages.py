import argparse
import imutils
import cv2
import os
import time

from src.helpers import find_files_in_dir


def augment(folder: str):
    start = time.time()
    image_files = find_files_in_dir(folder)
    angles = [-2, -1, 1, 2]

    for image in image_files:
        # skip non-image files
        if not os.path.basename(image).split('.')[1] in ['png', 'jpg', 'jpeg']:
            continue

        im = cv2.imread(image)
        for angle in angles:
            rotated = imutils.rotate(im, angle)
            rotated_image_name = os.path.basename(image).split('.')[0] + "_r_" + str(angle) + "." + \
                                 os.path.basename(image).split('.')[1]
            cv2.imwrite(os.path.dirname(image) + "/" + rotated_image_name, rotated)
    end = time.time()
    print('Finished ! Total elapsed time:', end - start)

def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to the image directory")
    args = vars(ap.parse_args())
    augment(args.images)


if __name__ == "__main__":
    main()