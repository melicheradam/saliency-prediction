"""
Example usage of this script: python2.7 evaluate_results.py -gt ./demo/gt -sal ./demo/sal
"""
import json
import salience_metrics as sm
import os
import argparse
import cv2
from statistics import mean
import multiprocessing as mp
import tqdm
from helpers import fix_consistency
import config
import numpy as np

AUC_WARNING_TRESHOLD = 0.75
NUM_OF_PARALLEL_PROCESSES = 6
FORCED_IMAGE_RESIZE_DIMS = (640, 360)

def resize_images(gt, gt_raw, s_map, generalized_map, pred_generalized_map, force=False):

    if not gt.shape == gt_raw.shape == s_map.shape == generalized_map.shape == pred_generalized_map:
        min_height = min(gt.shape[0], gt_raw.shape[0], s_map.shape[0], generalized_map.shape[0], pred_generalized_map.shape[0])
        min_width = min(gt.shape[1], gt_raw.shape[1], s_map.shape[1], generalized_map.shape[1], pred_generalized_map.shape[1])
        gt = cv2.resize(gt, (min_width, min_height))
        gt_raw = cv2.resize(gt_raw, (min_width, min_height))
        s_map = cv2.resize(s_map, (min_width, min_height))
        generalized_map = cv2.resize(generalized_map, (min_width, min_height))
        pred_generalized_map = cv2.resize(pred_generalized_map, (min_width, min_height))

    elif force:
        gt = cv2.resize(gt, FORCED_IMAGE_RESIZE_DIMS)
        gt_raw = cv2.resize(gt_raw, FORCED_IMAGE_RESIZE_DIMS)
        s_map = cv2.resize(s_map, FORCED_IMAGE_RESIZE_DIMS)
        generalized_map = cv2.resize(generalized_map, FORCED_IMAGE_RESIZE_DIMS)
        pred_generalized_map = cv2.resize(pred_generalized_map, FORCED_IMAGE_RESIZE_DIMS)

    return gt, gt_raw, s_map, generalized_map, pred_generalized_map


# Count all saliency metrics for given groundtruth and predicted saliency map
def calculate_metrics(pack):
    gt_path = pack[1][0]  # GT path
    gt_raw_path = pack[1][1]  # Binary GT path
    sal_path = pack[1][2]  # Saliency map path
    gt_generalized_path = pack[1][3]  # Generalized GT path
    pred_generalized_path = pack[1][4] # Saliency map predicted by the generalized model
    config = pack[2]  # Object holding true/false values for metric names
    gt_raw_all = pack[3]  # All binary GTs

    gt = cv2.imread(gt_path, 0)  # Grayscale blurred image (0-255)
    gt_raw = cv2.imread(gt_raw_path, 0)  # Grayscale binary image (0-255)
    s_map = cv2.imread(sal_path, 0)  # Grayscale blurred image (0-255)
    gt_generalized_map = cv2.imread(gt_generalized_path, 0)  # Grayscale blurred image (0-255)
    pred_generalized_map = cv2.imread(pred_generalized_path, 0)  # Grayscale blurred image (0-255)

    gt, gt_raw, s_map, gt_generalized_map, pred_generalized_map = resize_images(gt, gt_raw, s_map, gt_generalized_map, pred_generalized_map)

    s_map_norm = sm.normalize_map(s_map)
    gt_generalized_path_norm = sm.normalize_map(gt_generalized_map)
    pred_generalized_path_norm = sm.normalize_map(pred_generalized_map)

    result = {}

    if config["AUC-Judd"]:
        auc_judd_score = sm.auc_judd(s_map_norm, gt_raw)
        result["AUC-Judd"] = auc_judd_score
        #print('auc judd (1 = best):', auc_judd_score)

    if config["AUC-Borji"]:
        auc_borji_score = sm.auc_borji(s_map_norm, gt_raw)
        result["AUC-Borji"] = auc_borji_score
        #print('auc borji (1 = best):', auc_borji_score)

    if config["AUC-Shuff"]:
        auc_shuff_score = sm.auc_shuff(s_map_norm, gt_raw, gt_raw_all)
        result["AUC-Shuff"] = auc_shuff_score
        #print('auc shuffled (1 = best):', auc_shuff_score)

    if config["NSS"]:
        nss_score = sm.nss(s_map, gt_raw)
        result["NSS"] = nss_score
        #print('nss (<2,5; 4> = best):', nss_score)

    if config["IG"]:
        infogain_score = sm.infogain(s_map_norm, gt_raw, pred_generalized_path_norm)
        result["IG"] = infogain_score
        #print('info gain (~5 = best):', infogain_score)

    if config["SIM"]:
        sim_score = sm.similarity(s_map, gt)
        result["SIM"] = sim_score
        #print('sim score (1 = best):', sim_score)

    if config["CC"]:
        cc_score = sm.cc(s_map, gt)
        result["CC"] = cc_score
        #print('cc score (1 = best):',cc_score)

    if config["KLdiv"]:
        kldiv_score = sm.kldiv(s_map, gt)
        result["KLdiv"] = kldiv_score
        #print('kldiv score (0 = best):',kldiv_score)

    return result


def evaluate_model(gt_directory, gt_raw_directory, sm_directory, pred_gen_directory, output_file=None, config=config):

    # Find paths of all ground-truth and predicted maps and sort them
    gt_dir_path = os.path.join(gt_directory)
    gt_raw_dir_path = os.path.join(gt_raw_directory)
    gt_generalized_dir_path = os.path.join(pred_gen_directory)
    sm_dir_path = os.path.join(sm_directory)
    pred_gen_dir_path = os.path.join(pred_gen_directory)

    gt_files = os.listdir(gt_dir_path)
    gt_raw_files = os.listdir(gt_raw_dir_path)
    sm_files = os.listdir(sm_dir_path)
    gt_generalized_files = os.listdir(gt_generalized_dir_path)
    pred_gen_files = os.listdir(pred_gen_dir_path)

    print(len(gt_files), len(gt_raw_files), len(sm_files), len(gt_generalized_files), len(pred_gen_files))


    gt_files, sm_files = fix_consistency(gt_files, sm_files)
    gt_files, gt_raw_files = fix_consistency(gt_files, gt_raw_files)
    gt_files, gt_generalized_files = fix_consistency(gt_files, gt_generalized_files)
    gt_files, pred_gen_files = fix_consistency(gt_files, pred_gen_files)

    gt_files_sorted = sort_by_name(gt_files)
    gt_raw_files_sorted = sort_by_name(gt_raw_files)
    sm_files_sorted = sort_by_name(sm_files)
    gt_generalized_files_sorted = sort_by_name(gt_generalized_files)
    pred_gen_files_sorted = sort_by_name(pred_gen_files)

    gt_paths = [os.path.join(gt_directory, file) for file in gt_files_sorted]
    gt_raw_paths = [os.path.join(gt_raw_directory, file) for file in gt_raw_files_sorted]
    gt_generalized_paths = [os.path.join(pred_gen_directory, file) for file in gt_generalized_files_sorted]
    sm_paths = [os.path.join(sm_directory, file) for file in sm_files_sorted]
    pred_gen_paths = [os.path.join(pred_gen_directory, file) for file in pred_gen_files_sorted]

    assert len(gt_paths) == len(gt_raw_paths) == len(sm_paths) == len(gt_generalized_paths) == len(pred_gen_paths)

    total_num_of_files = len(gt_files_sorted)
    print("Total number of files to evaluate: ", total_num_of_files)

    pool = mp.Pool(NUM_OF_PARALLEL_PROCESSES)

    results = list(
        tqdm.tqdm(
            pool.imap(
                calculate_metrics,
                [(i, (a, b, c, d, e), config.EVALUATION_METRICS, gt_raw_paths) for (i, (a, b, c, d, e)) in enumerate(zip(gt_paths, gt_raw_paths, sm_paths, gt_generalized_paths, pred_gen_paths))]
            ),
            total=total_num_of_files
        )
    )

    auc_judd_scores, auc_borji_scores, auc_shuff_scores, nss_scores, infogain_scores, sim_scores, cc_scores, kldiv_scores  = [], [], [], [], [], [], [], []

    for res in results:
        if config.EVALUATION_METRICS["AUC-Judd"]:
            auc_judd_scores.append(res["AUC-Judd"])
        if config.EVALUATION_METRICS["AUC-Borji"]:
            auc_borji_scores.append(res["AUC-Borji"])
        if config.EVALUATION_METRICS["AUC-Shuff"]:
            auc_shuff_scores.append(res["AUC-Shuff"])
        if config.EVALUATION_METRICS["NSS"]:
            nss_scores.append(res["NSS"])
        if config.EVALUATION_METRICS["IG"]:
            infogain_scores.append(res["IG"])
        if config.EVALUATION_METRICS["SIM"]:
            sim_scores.append(res["SIM"])
        if config.EVALUATION_METRICS["CC"]:
            cc_scores.append(res["CC"])
        if config.EVALUATION_METRICS["KLdiv"]:
            kldiv_scores.append(res["KLdiv"])

    print("Overall metrics are:")
    overall_results = {}

    if config.EVALUATION_METRICS["AUC-Judd"]:

        for i, score in enumerate(auc_judd_scores):
            if score <= AUC_WARNING_TRESHOLD:
                print("Weak prediction (AUC-Judd = ", score, ") for image:", os.path.basename(gt_paths[i]))

        auc_judd_mean = mean(auc_judd_scores)
        overall_results["AUC-Judd"] = auc_judd_mean
        print('auc judd (1 = best):', auc_judd_mean)

    if config.EVALUATION_METRICS["AUC-Borji"]:
        auc_borji_mean = mean(auc_borji_scores)
        overall_results["AUC-Borji"] = auc_borji_mean
        print('auc borji (1 = best):', auc_borji_mean)

    if config.EVALUATION_METRICS["AUC-Shuff"]:
        auc_shuff_mean = mean(auc_shuff_scores)
        overall_results["AUC-Shuff"] = auc_shuff_mean
        print('auc shuffled (1 = best):', auc_shuff_mean)

    if config.EVALUATION_METRICS["NSS"]:
        nss_mean = mean(nss_scores)
        overall_results["NSS"] = nss_mean
        print('nss (<2,5; 4> = best):', nss_mean)

    if config.EVALUATION_METRICS["IG"]:
        infogain_mean = mean(infogain_scores)
        overall_results["IG"] = infogain_mean
        print('info gain (~5 = best):', infogain_mean)

    if config.EVALUATION_METRICS["SIM"]:
        sim_mean = mean(sim_scores)
        overall_results["SIM"] = sim_mean
        print('sim score (1 = best):', sim_mean)

    if config.EVALUATION_METRICS["CC"]:
        cc_mean = mean(cc_scores)
        overall_results["CC"] = cc_mean
        print('cc score (1 = best):', cc_mean)

    if config.EVALUATION_METRICS["KLdiv"]:
        kldiv_mean = mean(kldiv_scores)
        overall_results["KLdiv"] = kldiv_mean
        print('kldiv score (0 = best):', kldiv_mean)

    if output_file:
        with open(output_file, 'w') as outfile:
            json.dump(overall_results, outfile)

    return overall_results


def sort_by_name(maps):
    return sorted(maps, key=lambda i: os.path.splitext(os.path.basename(i))[0])


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    import time
    start = time.time()

    print("Please make sure you're running the metric evaluation under python2.7")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-gt", "--ground-truth", metavar="GT",
                        help="path for groundtruth saliency maps directory")

    parser.add_argument("-bin", "--binary-fixations", metavar="BN",
                        help="path for binary maps")

    parser.add_argument("-sal", "--saliency-maps", metavar="SAL",
                        help="path for result saliency maps directory")

    parser.add_argument("-output", "--output-file", metavar="OUT",
                        help="csv file path where results will be stored")

    parser.add_argument("-pg", "--predicted-generalized", metavar="OUT",
                        help="saliency maps predicted by the generalized model")

    args = parser.parse_args()
    evaluate_model(args.ground_truth, args.binary_fixations, args.saliency_maps, args.predicted_generalized, args.output_file)

    end = time.time()

    print('Total elapsed time:', end - start)

if __name__ == "__main__":
    main()
