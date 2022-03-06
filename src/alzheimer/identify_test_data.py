# Find stimuli which are present for all observers and print them
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from helpers import fixpos_to_densemap, find_files_in_dir

fixations_dir = "data/PSD/fixations/Personalized_maps"
subject_dirs = [f.path for f in os.scandir(fixations_dir) if f.is_dir()]
unique_maps = {}

for subdir in subject_dirs:
    subject_files = set([os.path.basename(x) for x in find_files_in_dir(subdir)])
    for file in subject_files:
        if file in unique_maps:
            unique_maps[file] = unique_maps[file] + 1
        else:
            unique_maps[file] = 1


print(unique_maps)