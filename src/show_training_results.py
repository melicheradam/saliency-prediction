import argparse
import os
import shutil
from pathlib import Path

from helpers import find_files_in_dir
from scipy.stats import shapiro
import json
import numpy as np
import matplotlib
import seaborn as sns
import config
matplotlib.use('Agg')


def plot_distribution(label_x, data, output_path, label_y='Number of observers',):
    plot = sns.displot(data, kind="hist", kde=True)
    plot.set(title=label_x, ylabel=label_y)
    plot.savefig(output_path + '/' + label_x + '.png')


def print_shapiro(data):
    stat, p = shapiro(data)
    print('Shapiro-Wilk test: Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian')
    else:
        print('Sample does not look Gaussian')

def print_overall_scores(results, plot=False):

    auc_judd_values, auc_borji_values, auc_shuff_values, nss_values, ig_values, sim_values, cc_values, kldiv_values = [], [], [], [], [], [], [], []

    for res in results:
        with open(res) as f:
            res_data = json.load(f)

        if config.EVALUATION_METRICS["AUC-Judd"]:
            auc_judd = res_data["AUC-Judd"]
            auc_judd_values.append(auc_judd)

        if config.EVALUATION_METRICS["AUC-Borji"]:
            auc_borji = res_data["AUC-Borji"]
            auc_borji_values.append(auc_borji)

        if config.EVALUATION_METRICS["AUC-Shuff"]:
            auc_shuff = res_data["AUC-Shuff"]
            auc_shuff_values.append(auc_shuff)

        if config.EVALUATION_METRICS["NSS"]:
            nss = res_data["NSS"]
            nss_values.append(nss)

        if config.EVALUATION_METRICS["IG"]:
            ig = res_data["IG"]
            ig_values.append(ig)

        if config.EVALUATION_METRICS["SIM"]:
            sim = res_data["SIM"]
            sim_values.append(sim)

        if config.EVALUATION_METRICS["CC"]:
            cc = res_data["CC"]
            cc_values.append(cc)

        if config.EVALUATION_METRICS["KLdiv"]:
            kldiv = res_data["KLdiv"]
            kldiv_values.append(kldiv)

    if config.EVALUATION_METRICS["AUC-Judd"]:
        print('auc judd (1 = best):', np.mean(auc_judd_values))
    if config.EVALUATION_METRICS["AUC-Borji"]:
        print('auc borji (1 = best):', np.mean(auc_borji_values))
    if config.EVALUATION_METRICS["AUC-Shuff"]:
        print('auc shuffled (1 = best):', np.mean(auc_shuff_values))
    if config.EVALUATION_METRICS["NSS"]:
        print('nss (<2,5; 4> = best):', np.mean(nss_values))
    if config.EVALUATION_METRICS["IG"]:
        print('info gain (~5 = best):', np.mean(ig_values))
    if config.EVALUATION_METRICS["SIM"]:
        print('sim score (1 = best):', np.mean(sim_values))
    if config.EVALUATION_METRICS["CC"]:
        print('cc score (1 = best):', np.mean(cc_values))
    if config.EVALUATION_METRICS["KLdiv"]:
        print('kldiv score (0 = best):', np.mean(kldiv_values))

    if plot:
        output_path = 'test-results/distributions-validation-set'
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

        plot_distribution(label_x='AUC-judd distribution', data=auc_judd_values, output_path=output_path)
        plot_distribution(label_x='AUC-borji distribution', data=auc_borji_values, output_path=output_path)
        plot_distribution(label_x='AUC-shuff distribution', data=auc_shuff_values, output_path=output_path)
        plot_distribution(label_x='NSS distribution', data=nss_values, output_path=output_path)
        plot_distribution(label_x='IG distribution', data=ig_values, output_path=output_path)
        plot_distribution(label_x='SIM distribution', data=sim_values, output_path=output_path)
        plot_distribution(label_x='CC distribution', data=cc_values, output_path=output_path)
        plot_distribution(label_x='KL-div distribution', data=kldiv_values, output_path=output_path)

def print_individual_scores(results):

    for res in results:
        with open(res) as f:
            res_data = json.load(f)

        print(res)

        if config.EVALUATION_METRICS["AUC-Judd"]:
            auc_judd = res_data["AUC-Judd"]
            print('auc judd (1 = best):', auc_judd)

        if config.EVALUATION_METRICS["AUC-Borji"]:
            auc_borji = res_data["AUC-Borji"]
            print('auc borji (1 = best):', auc_borji)

        if config.EVALUATION_METRICS["AUC-Shuff"]:
            auc_shuff = res_data["AUC-Shuff"]
            print('auc shuffled (1 = best):', auc_shuff)

        if config.EVALUATION_METRICS["NSS"]:
            nss = res_data["NSS"]
            print('nss (<2,5; 4> = best):', nss)

        if config.EVALUATION_METRICS["IG"]:
            ig = res_data["IG"]
            print('info gain (~5 = best):', ig, 2)

        if config.EVALUATION_METRICS["SIM"]:
            sim = res_data["SIM"]
            print('sim score (1 = best):', sim, 2)

        if config.EVALUATION_METRICS["CC"]:
            cc = res_data["CC"]
            print('cc score (1 = best):', cc)

        if config.EVALUATION_METRICS["KLdiv"]:
            kldiv = res_data["KLdiv"]
            print('kldiv score (0 = best):', kldiv)


# e.g res_a = generalized result, res_b = personalized results
# if personalized score is better, AUC difference is positive
def print_comparison(res_a, res_b, plot=False):
    auc_judd_values, auc_borji_values, auc_shuff_values, nss_values, ig_values, sim_values, cc_values, kldiv_values = [], [], [], [], [], [], [], []

    for res in zip(res_a, res_b):
        with open(res[0]) as f:
            res_a_data = json.load(f)

        with open(res[1]) as f:
            res_b_data = json.load(f)

        print("Comparing " + res[0] + " and " + res[1])

        if config.EVALUATION_METRICS["AUC-Judd"]:
            auc_judd = res_b_data["AUC-Judd"] - res_a_data["AUC-Judd"]
            auc_judd_values.append(auc_judd)
            print('auc judd (1 = best):', auc_judd)

        if config.EVALUATION_METRICS["AUC-Borji"]:
            auc_borji = res_b_data["AUC-Borji"] - res_a_data["AUC-Borji"]
            auc_borji_values.append(auc_borji)
            print('auc borji (1 = best):', auc_borji)

        if config.EVALUATION_METRICS["AUC-Shuff"]:
            auc_shuff = res_b_data["AUC-Shuff"] - res_a_data["AUC-Shuff"]
            auc_shuff_values.append(auc_shuff)
            print('auc shuffled (1 = best):', auc_shuff)

        if config.EVALUATION_METRICS["NSS"]:
            nss = res_b_data["NSS"] - res_a_data["NSS"]
            nss_values.append(nss)
            print('nss (<2,5; 4> = best):', nss)

        if config.EVALUATION_METRICS["IG"]:
            ig = res_b_data["IG"] - res_a_data["IG"]
            ig_values.append(ig)
            print('info gain (~5 = best):', ig)

        if config.EVALUATION_METRICS["SIM"]:
            sim = res_b_data["SIM"] - res_a_data["SIM"]
            sim_values.append(sim)
            print('sim score (1 = best):', sim)

        if config.EVALUATION_METRICS["CC"]:
            cc = res_b_data["CC"] - res_a_data["CC"]
            cc_values.append(cc)
            print('cc score (1 = best):', cc)

        if config.EVALUATION_METRICS["KLdiv"]:
            kldiv = res_b_data["KLdiv"] - res_a_data["KLdiv"]
            kldiv_values.append(kldiv)
            print('kldiv score (0 = best):', kldiv)

    print("Average difference:")
    if config.EVALUATION_METRICS["AUC-Judd"]:
        print('auc judd (1 = best):', np.mean(auc_judd_values))
        print('stdev: ', np.std(auc_judd_values))
        print_shapiro(auc_judd_values)

    if config.EVALUATION_METRICS["AUC-Borji"]:
        print('auc borji (1 = best):', np.mean(auc_borji_values))
        print('stdev: ', np.std(auc_borji_values))
        print_shapiro(auc_borji_values)

    if config.EVALUATION_METRICS["AUC-Shuff"]:
        print('auc shuffled (1 = best):', np.mean(auc_shuff_values))
        print('stdev: ', np.std(auc_shuff_values))
        print_shapiro(auc_shuff_values)

    if config.EVALUATION_METRICS["NSS"]:
        print('nss (<2,5; 4> = best):', np.mean(nss_values))
        print('stdev: ', np.std(nss_values))
        print_shapiro(nss_values)

    if config.EVALUATION_METRICS["IG"]:
        print('info gain (~5 = best):', np.mean(ig_values))
        print('stdev: ', np.std(ig_values))
        print_shapiro(ig_values)

    if config.EVALUATION_METRICS["SIM"]:
        print('sim score (1 = best):', np.mean(sim_values))
        print('stdev: ', np.std(sim_values))
        print_shapiro(sim_values)

    if config.EVALUATION_METRICS["CC"]:
        print('cc score (1 = best):', np.mean(cc_values))
        print('stdev: ', np.std(cc_values))
        print_shapiro(cc_values)

    if config.EVALUATION_METRICS["KLdiv"]:
        print('kldiv score (0 = best):', np.mean(kldiv_values))
        print('stdev: ', np.std(kldiv_values))
        print_shapiro(kldiv_values)

    if plot:
        output_path = 'test-results/comparison-validation-set'
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

        plot_distribution(label_x='AUC-judd improvement', data=auc_judd_values, output_path=output_path)
        plot_distribution(label_x='AUC-borji improvement', data=auc_borji_values, output_path=output_path)
        plot_distribution(label_x='AUC-shuff improvement', data=auc_shuff_values, output_path=output_path)
        plot_distribution(label_x='NSS improvement', data=nss_values, output_path=output_path)
        plot_distribution(label_x='IG improvement', data=ig_values, output_path=output_path)
        plot_distribution(label_x='SIM improvement', data=sim_values, output_path=output_path)
        plot_distribution(label_x='CC improvement', data=cc_values, output_path=output_path)
        plot_distribution(label_x='KL-div improvement', data=kldiv_values, output_path=output_path)


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-name", "--model-name", metavar="MN",
                        help="name of the model which was evaluated")

    parser.add_argument("-res", "--results-dir", metavar="RD",
                        help="path to results directory")

    args = parser.parse_args()
    res_dir = Path(args.results_dir)
    result_files = []
    for item in os.listdir(res_dir):
        if args.model_name in str(item):
            result_files += find_files_in_dir(os.path.join(res_dir, item), filenameContains='json')

    print("Overall score of the personalized model:")
    print_overall_scores(result_files, plot=True)

    print("Individual scores of the personalized model:")
    print_individual_scores(result_files)


if __name__ == "__main__":
    main()


