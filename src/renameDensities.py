# Purpose of this script is to rename density map files from the personalized dataset from "d1.png" to "1.png"
# So that they match the naming convention used in other saliency maps

import argparse
import os
from helpers import find_files_in_dir


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the image directory")
args = vars(ap.parse_args())

image_files = find_files_in_dir(args["images"])

for image in image_files:
    # skip non-image files
    if not os.path.basename(image).split('.')[1] in ['png', 'jpg', 'jpeg']:
        continue

    new_image_name = os.path.basename(image).split('.')[0].split('d')[1]
    print("New image name: " + new_image_name)
    os.rename(image, os.path.dirname(image) + "/" + new_image_name + "." + os.path.basename(image).split('.')[1])
