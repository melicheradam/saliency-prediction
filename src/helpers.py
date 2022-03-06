import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import shutil


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


def delete_dir_with_content(path):
    shutil.rmtree(path)

"""
Parse a .fix file and return the extracted info
__________________________________________________
Fixation Listing: (fixation number, x position, y position, begin time, end time, duration)
1. 449, 270, 0.150, 0.430, 0.280
2. 361, 156, 0.500, 0.791, 0.291
3. 566, 556, 1.001, 1.231, 0.230
4. 400, 548, 1.291, 1.562, 0.271
5. 387, 619, 1.592, 1.792, 0.200
6. 698, 672, 1.892, 2.093, 0.201
7. 730, 528, 2.133, 2.493, 0.360
8. 719, 288, 2.663, 3.094, 0.431
9. 805, 295, 3.134, 3.535, 0.401
10. 451, 287, 3.635, 3.935, 0.300
"""


def parse_fixation_file(file_path, merge_fixations=False):
    import re
    f = open(file_path, "r")
    fixations = []

    for line in f:
        if re.match(r"^\d+\..*$", line):
            values = line.replace("\n", "").replace(",", "").split(" ")
            fixations.append([float(values[1]), float(values[2]), float(values[5])])
    f.close()

    # Find fixations which are closer than 100px from each other
    if merge_fixations:
        return merge_close_fixations(fixations)
    else:
        return fixations

def gaussian_mask(sizex, sizey, sigma=33, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def fixpos_to_densemap(fix_arr, width, height, imgfile, alpha=0.5, threshold=10, colormap=cv2.COLORMAP_BONE):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap
    """

    heatmap = np.zeros((height, width), np.float32)
    for fixation_index in range(fix_arr.shape[0]):
        heatmap += gaussian_mask(width, height, 33, (fix_arr[fixation_index, 0], fix_arr[fixation_index, 1]),
                          
                                 fix_arr[fixation_index, 2])

    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255.0
    heatmap = heatmap.astype("uint8")
    heatmap = cv2.GaussianBlur(heatmap, (327, 327), 0)

    heatmap = heatmap.astype("float32")
    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255
    heatmap = heatmap.astype("uint8")

    if imgfile.any():
        # Resize heatmap to imgfile shape
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, colormap)

        # Create mask
        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile * mask + heatmap_color * (1 - mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1 - alpha, marge, alpha, 0)

        return marge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        return heatmap


def merge_close_fixations(fixation_arr):
    merged_fixations = fixation_arr
    while len(merged_fixations) > 1:
        i_index = 0
        merged = False
        for i in merged_fixations:
            j_index = 0
            for j in merged_fixations:
                if i_index < j_index:
                    p1 = np.array((i[0], i[1]))
                    p2 = np.array((j[0], j[1]))
                    dist = np.sqrt(((p1 - p2) ** 2).sum())
                    if dist <= 100:
                        p_merged = (p1 + p2) / 2
                        merged_fixations = [x for x in merged_fixations if
                                            json.dumps(x) != json.dumps(i) and json.dumps(x) != json.dumps(j)]
                        merged_fixations.append([p_merged[0], p_merged[1], i[2] + j[2]])
                        merged = True
                        break

                j_index += 1
            if merged:
                break
            i_index += 1

        # If no more fixations need to be merged
        if not merged:
            break

    return merged_fixations


def fix_consistency(filepaths_a, filepaths_b):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps."""

    filenames_a = [os.path.splitext(file)[0] for file in filepaths_a]
    filenames_b = [os.path.splitext(file)[0] for file in filepaths_b]

    matching_files = list(set(filenames_a) & set(filenames_b))

    filtered_a = [path for path in filepaths_a if os.path.splitext(path)[0] in matching_files]
    filtered_b = [path for path in filepaths_b if os.path.splitext(path)[0] in matching_files]

    if abs(len(filtered_a) - len(filepaths_a)):
        print("Filepath mismatch of " + str(abs(len(filtered_a) - len(filepaths_a))) + " files !")

        #mismatched_files = [path for path in filepaths_a if os.path.basename(path).split('.')[0] not in matching_files]
        #print("Mismatched files:")
        #print(mismatched_files)

    elif abs(len(filtered_b) - len(filepaths_b)):
        print("Filepath mismatch of " + str(abs(len(filtered_b) - len(filepaths_b))) + " files !")

        #mismatched_files = [path for path in filepaths_b if os.path.basename(path).split('.')[0] not in matching_files]
        #print("Mismatched files:")
        #print(mismatched_files)

    return filtered_a, filtered_b

