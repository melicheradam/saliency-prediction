"""
Script is used to copy generalized saliency maps before training of generalized model.
Only maps which are also available for the given observer are copied.
Example usage of this script: python3 prepareData.py -genpath eyetrackingdata/fixdens/Generated_Generalized_maps -perspath eyetrackingdata/fixdens/Generated_Personalized_maps/alex/SM -targetpath encoder-decoder-model/data/personalized/saliency
"""

import os
import argparse
from helpers import fix_consistency
import shutil


def copy(generalized_directory, personalized_directory, target_directory):

    generalized_dir_path = os.path.join(generalized_directory)
    personalized_dir_path = os.path.join(personalized_directory)
    generalized_files = os.listdir(generalized_dir_path)
    personalized_files = os.listdir(personalized_dir_path)

    generalized_files, personalized_files = fix_consistency(generalized_files, personalized_files)
    generalized_files_sorted = sorted(generalized_files, key=lambda i: os.path.splitext(os.path.basename(i))[0])

    shutil.rmtree(target_directory)
    os.mkdir(target_directory)

    for file in generalized_files_sorted:
        shutil.copyfile(os.path.join(generalized_directory, file), os.path.join(target_directory, file))

    print("Number of copied files: ", len(generalized_files_sorted))


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-genpath", "--generalized-path", metavar="GP",
                        help="path for generalized saliency maps directory")

    parser.add_argument("-perspath", "--personalized-path", metavar="PP",
                        help="path for saliency maps of particular observer")

    parser.add_argument("-targetpath", "--target-path", metavar="TP",
                        help="path, where resulting files will be copied")

    args = parser.parse_args()
    copy(args.generalized_path, args.personalized_path, args.target_path)


if __name__ == "__main__":
    main()