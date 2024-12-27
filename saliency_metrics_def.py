import numpy as np
import random
import math


def generate_dummy(size=14, num_fixations=100, num_salience_points=200):
    # Generate dummy ground truth and salience map
    discrete_gt = np.zeros((size, size))
    s_map = np.zeros((size, size))

    for _ in range(num_fixations):
        discrete_gt[np.random.randint(size), np.random.randint(size)] = 1.0

    for _ in range(num_salience_points):
        s_map[np.random.randint(size), np.random.randint(size)] = 255 * round(random.random(), 1)

    # Check if gt and s_map are the same size
    assert discrete_gt.shape == s_map.shape, 'Ground truth and salience map sizes do not match'
    return s_map, discrete_gt


def normalize_map(s_map):
    # Normalize the salience map
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
    return norm_s_map


def discretize_gt(gt):
    import warnings
    warnings.warn('The GT discretization can be improved')
    return gt / 255


def auc_judd(s_map, gt):
    gt = discretize_gt(gt)

    # Print shapes of s_map and gt
    print("Shape of s_map:", s_map.shape)
    print("Shape of gt:", gt.shape)
    #fine mie modifiche
    
    thresholds = [s_map[i][k] for i in range(gt.shape[0]) for k in range(gt.shape[1]) if gt[i][k] > 0]

    num_fixations = np.sum(gt)
    thresholds = sorted(set(thresholds))

    area = [(0.0, 0.0)]
    for thresh in thresholds:
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)
        fp = (np.sum(temp) - num_overlap) / ((gt.size) - num_fixations)
        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_borji(s_map, gt, splits=100, stepsize=0.1):
    gt = discretize_gt(gt)
    num_fixations = int(np.sum(gt))
    print("in auc_borji, num_fixations: ", num_fixations)
    num_pixels = s_map.shape[0] * s_map.shape[1]

    random_numbers = [[np.random.randint(num_pixels) for _ in range(num_fixations)] for _ in range(splits)]

    aucs = []
    for indices in random_numbers:
        r_sal_map = [s_map[k % s_map.shape[0] - 1, k // s_map.shape[0]] for k in indices]
        thresholds = sorted(set([0.1 * i for i in range(1, 10)]))

        r_sal_map = np.array(r_sal_map)
        area = [(0.0, 0.0)]
        for thresh in thresholds:
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / num_fixations
            fp = len(r_sal_map[r_sal_map > thresh]) / num_fixations
            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def similarity(s_map, gt):
    s_map = normalize_map(s_map)
    gt = normalize_map(gt)
    s_map /= np.sum(s_map)
    gt /= np.sum(gt)
    sim = sum(min(gt[i[0], i[1]], s_map[i[0], i[1]]) for i in zip(*np.where(gt > 0)))
    return sim


def cc(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    r = (s_map_norm * gt_norm).sum() / math.sqrt((s_map_norm ** 2).sum() * (gt_norm ** 2).sum())
    return r


def kldiv(s_map, gt):
    # Ensure s_map and gt are of type float64
    s_map = s_map.astype(np.float64)
    gt = gt.astype(np.float64)
    #fine mie modifiche
    
    s_map /= np.sum(s_map)
    gt /= np.sum(gt)
    eps = 2.2204e-16
    return np.sum(gt * np.log(eps + gt / (s_map + eps)))


