"""
Please make sure you're running the metric evaluation under python2.7
"""
import cv2
import numpy as np
import random
import math


def generate_dummy(size=14, num_fixations=100, num_salience_points=200):
	# first generate dummy gt and salience map
	discrete_gt = np.zeros((size, size))
	s_map = np.zeros((size, size))

	for i in range(0, num_fixations):
		discrete_gt[np.random.randint(size), np.random.randint(size)] = 1.0

	for i in range(0, num_salience_points):
		s_map[np.random.randint(size), np.random.randint(size)] = 255 * round(random.random(), 1)

	# check if gt and s_map are same size
	assert discrete_gt.shape == s_map.shape, 'sizes of ground truth and salience map don\'t match'
	return s_map, discrete_gt


def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	min_smap = np.min(s_map)
	norm_s_map = (s_map - min_smap) / ((np.max(s_map) - min_smap) * 1.0)
	return norm_s_map


def discretize_gt(gt, python3=0):
	if python3:
		return gt // 255
	else:
		return gt / 255


def auc_judd(s_map, gt, python3=0):
	# ground truth is discrete, s_map is continous and normalized
	gt = discretize_gt(gt, python3)

	# thresholds are calculated from the salience map, only at places where fixations are present
	thresholds = [s_map[i][k] for i in range(0, gt.shape[0]) for k in range(0, gt.shape[1]) if gt[i][k] > 0]

	num_fixations = np.sum(gt)
	# num fixations is no. of salience map values at gt >0

	thresholds = sorted(set(thresholds))

	# fp_list = []
	# tp_list = []
	area = []
	area.append((0.0, 0.0))
	for thresh in thresholds:
		# in the salience map, keep only those pixels with values above threshold
		temp = np.zeros(s_map.shape)
		temp[s_map >= thresh] = 1.0
		assert np.max(gt) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
		assert np.max(s_map) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
		num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
		tp = num_overlap / (num_fixations * 1.0)

		# total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
		# this becomes nan when gt is full of fixations..this won't happen
		fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

		area.append((round(tp, 4), round(fp, 4)))
	# tp_list.append(tp)
	# fp_list.append(fp)

	# tp_list.reverse()
	# fp_list.reverse()
	area.append((1.0, 1.0))
	# tp_list.append(1.0)
	# fp_list.append(1.0)
	# print tp_list
	area.sort(key=lambda x: x[0])
	tp_list = [x[0] for x in area]
	fp_list = [x[1] for x in area]
	return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_borji(s_map, gt, splits=100, stepsize=0.1):
	gt = discretize_gt(gt)
	num_fixations = np.sum(gt)

	num_pixels = s_map.shape[0] * s_map.shape[1]
	random_numbers = []
	for i in range(0, splits):
		temp_list = []
		for k in range(0, num_fixations):
			temp_list.append(np.random.randint(num_pixels))
		random_numbers.append(temp_list)

	aucs = []
	# in these values, we need to find thresholds and calculate auc
	thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# for each split, calculate auc
	for i in random_numbers:
		r_sal_map = []
		for k in i:
			r_sal_map.append(s_map[k % s_map.shape[0] - 1, k / s_map.shape[0]])
		r_sal_map = np.array(r_sal_map)

		area = []
		area.append((0.0, 0.0))
		for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
			temp = np.zeros(s_map.shape)
			temp[s_map >= thresh] = 1.0
			num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
			tp = num_overlap / (num_fixations * 1.0)

			# fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
			fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

			area.append((round(tp, 4), round(fp, 4)))

		area.append((1.0, 1.0))
		area.sort(key=lambda x: x[0])
		tp_list = [x[0] for x in area]
		fp_list = [x[1] for x in area]

		aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

	return np.mean(aucs)


def auc_shuff(s_map, gt, gt_bin_all, splits=100, stepsize=0.1, num_other=10):

	gt = discretize_gt(gt)
	num_fixations = np.sum(gt)

	# Collect all fixations from other maps
	union = np.zeros(gt.shape)
	for file in gt_bin_all[:num_other]:
		gt_bin = cv2.imread(file, 0)
		union = np.add(union, gt_bin)
		other_map = np.where(union > 0, 1, union)

	# Randomly choose N of these fixations
	x, y = np.where(other_map > 0)
	all_fixations_other = []
	for j in zip(x, y):
		all_fixations_other.append([j[0], j[1]])

	indices = np.arange(len(all_fixations_other), dtype=int)
	chosen_indices = np.random.choice(indices, num_fixations)

	other_map = np.zeros(gt.shape)
	for index in chosen_indices:
		other_map[all_fixations_other[index][0], all_fixations_other[index][1]] = 1

	x, y = np.where(other_map == 1)
	other_map_fixs = []
	for j in zip(x, y):
		other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
	ind = len(other_map_fixs)
	assert ind == np.sum(other_map), 'something is wrong in auc shuffle'
	num_fixations_other = min(ind, num_fixations)

	random_numbers = []
	for i in range(0, splits):
		temp_list = []
		t1 = np.random.permutation(ind)
		for k in t1:
			temp_list.append(other_map_fixs[k])
		random_numbers.append(temp_list)

	aucs = []
	# for each split, calculate auc
	for i in random_numbers:
		r_sal_map = []
		for k in i:
			r_sal_map.append(s_map[k % s_map.shape[0] - 1, k / s_map.shape[0]])
		r_sal_map = np.array(r_sal_map)

		# in these values, we need to find thresholds and calculate auc
		thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		thresholds = sorted(set(thresholds))

		area = []
		area.append((0.0, 0.0))
		for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
			temp = np.zeros(s_map.shape)
			temp[s_map >= thresh] = 1.0
			num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]

			tp = num_overlap / (num_fixations * 1.0)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
			fp = len(np.where(r_sal_map >= thresh)[0]) / (num_fixations_other * 1.0)

			area.append((round(tp, 4), round(fp, 4)))

		area.append((1.0, 1.0))
		area.sort(key=lambda x: x[0])
		tp_list = [x[0] for x in area]
		fp_list = [x[1] for x in area]

		aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

	return np.mean(aucs)


def nss(s_map, gt):
	gt = discretize_gt(gt)
	s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)

	x, y = np.where(gt == 1)
	temp = []
	for i in zip(x, y):
		temp.append(s_map_norm[i[0], i[1]])
	return np.mean(temp)


def infogain(s_map, gt, baseline_map):
	gt = discretize_gt(gt)
	# assuming s_map and baseline_map are normalized
	eps = 2.2204e-16

	s_map = s_map / (np.sum(s_map) * 1.0)
	baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

	# for all places where gt=1, calculate info gain
	temp = []
	x, y = np.where(gt == 1)
	for i in zip(x, y):
		temp.append(np.log2(eps + s_map[i[0], i[1]]) - np.log2(eps + baseline_map[i[0], i[1]]))

	return np.mean(temp)


def similarity(s_map, gt):
	# here gt is not discretized nor normalized
	s_map = normalize_map(s_map)
	gt = normalize_map(gt)
	s_map = s_map / (np.sum(s_map) * 1.0)
	gt = gt / (np.sum(gt) * 1.0)
	x, y = np.where(gt > 0)
	sim = 0.0
	for i in zip(x, y):
		sim = sim + min(gt[i[0], i[1]], s_map[i[0], i[1]])
	return sim


def cc(s_map, gt):
	s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
	gt_norm = (gt - np.mean(gt)) / np.std(gt)
	a = s_map_norm
	b = gt_norm
	r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum());
	return r


def kldiv(s_map, gt):
	s_map = s_map / (np.sum(s_map) * 1.0)
	gt = gt / (np.sum(gt) * 1.0)
	eps = 2.2204e-16
	return np.sum(gt * np.log(eps + gt / (s_map + eps)))




################################################
# img = cv2.imread('/home/tarun/mine/tensorflow_examples/image.jpg',0)
# print 'sim',similarity(img,img)
# print 'cc',cc(img,img)
# print 'kldiv',kldiv(img,img)
# import sys
# sys.exit(0)


#############################################

# Example usage
"""
gt = cv2.imread('./demo/gt/1.png',cv2.IMREAD_GRAYSCALE)
#cv2.imshow("Ground truth", gt)

s_map = cv2.imread('./demo/gt/1 copy.png',cv2.IMREAD_GRAYSCALE)
s_map_norm = normalize_map(s_map)

auc_judd_score = auc_judd(s_map_norm,gt)
print('auc judd (1 = best):', auc_judd_score)

auc_borji_score = auc_borji(s_map_norm,gt)
print('auc borji (1 = best):', auc_borji_score)

auc_shuff_score = auc_shuff(s_map_norm,gt,gt)
print('auc shuffled (1 = best):', auc_shuff_score)

nss_score = nss(s_map,gt)
print('nss (<2,5; 4> = best):', nss_score)

infogain_score = infogain(s_map_norm,gt,gt)
print('info gain (~5 = best):', infogain_score)

sim_score = similarity(s_map,gt)
print('sim score (1 = best):', sim_score)

cc_score = cc(s_map,gt)
print('cc score (1 = best):',cc_score)

kldiv_score = kldiv(s_map,gt)
print('kldiv score (0 = best):',kldiv_score)
"""