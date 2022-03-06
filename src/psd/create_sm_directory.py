import os
import argparse
import shutil


def create_sm_directories(fixations_dir):

    subdirs = [f.path for f in os.scandir(fixations_dir) if f.is_dir()]
    for subdir in subdirs:
        filenames = os.listdir(subdir)
        target = os.path.join(subdir, 'SM')
        os.mkdir(target)

        for filename in filenames:
            shutil.move(os.path.join(subdir, filename), target)

def create_personalized_directory(fixations_dir):

    personalized_directory_path = os.path.join(fixations_dir, 'Personalized_maps')

    if not os.path.exists(personalized_directory_path):
        os.mkdir(personalized_directory_path)

    subdirs = [f.path for f in os.scandir(fixations_dir) if f.is_dir()]
    for subdir in subdirs:
        if 'Sub_' in subdir:
            shutil.move(subdir, personalized_directory_path)

def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-fix", "--fixations-path", metavar="FP",
                        help="path for the fixations directory")

    args = parser.parse_args()
    create_sm_directories(args.fixations_path)
    create_personalized_directory(args.fixations_path)


if __name__ == "__main__":
    main()
