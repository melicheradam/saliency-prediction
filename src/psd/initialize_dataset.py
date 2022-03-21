import os
import argparse
import shutil
import scipy.io
import pandas as pd

### Had to copy this from src/helpers.py because import was not working
# Return files inside the provided directory and subdirectories
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
    # Skip hidden files and folders
    files = [f for f in files if not os.path.basename(f)[0] == '.']
    return files


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


# Convert .mat fixation files to .csv
def convert_mat_to_json(fixations_dir):

    # mat_files = find_files_in_dir(os.path.join(fixations_dir, 'Raw'))
    mat_files = find_files_in_dir(os.path.join(fixations_dir))

    for index, mat_file in enumerate(mat_files):
        if not os.path.splitext(mat_file)[1] == '.mat':
            continue

        content = scipy.io.loadmat(mat_file)
        mat = {k: v for k, v in content.items() if k[0] != '_'}
        print(mat)
        data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
        data.to_json(mat_file + '.json')
        print(index + 1, "/", len(mat_files))


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
    #create_sm_directories(args.fixations_path)
    #create_personalized_directory(args.fixations_path)
    convert_mat_to_json(args.fixations_path)


if __name__ == "__main__":
    main()
