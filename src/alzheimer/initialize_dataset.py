import argparse
import re
import shutil
import sys, os, inspect
import cv2
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from helpers import fixpos_to_densemap, find_files_in_dir

def rename_images(images_dir):
    all_images = find_files_in_dir(images_dir)

    for image in all_images:
        image_index = os.path.splitext(os.path.basename(image))[0]
        parent_dir = os.path.dirname(image)
        new_image_name = 'Part' + re.findall(r'\d+', os.path.basename(parent_dir))[0] + '-' + image_index + '.bmp'
        new_image_path = os.path.join(images_dir, new_image_name)
        print(new_image_path)
        shutil.move(image, new_image_path)

# Convert images from .bmp to .jpg
def convert_images(images_dir):
    all_images = find_files_in_dir(images_dir, filenameContains='.bmp')
    for image_path in all_images:
        image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(images_dir, os.path.splitext(os.path.basename(image_path))[0] + '.jpg'), image)
        os.remove(image_path)

def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-img", "--images-path", metavar="FP",
                        help="path for the images directory")

    args = parser.parse_args()
    rename_images(args.images_path)
    print("Dataset successfully initialized !")
    convert_images(args.images_path)
    print("Images successfully converted from .bmp to .jpg !")


if __name__ == "__main__":
    main()
