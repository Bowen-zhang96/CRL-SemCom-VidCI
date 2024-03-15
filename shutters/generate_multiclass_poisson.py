'''
Implements Multi-Class Poisson Disk Sampling from pseudocode in
Wei, Li-Yi. "Multi-class blue noise sampling." ACM Transactions on Graphics (TOG) 29.4 (2010): 1-8.

Written for square 2D images

Implemented by Cindy M. Nguyen
11/20/21
'''

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import torch

np.random.seed(123)


def build_r_matrix(r_set, n=2):
    # r_set is the user specified per-class values
    # r(k,j) specified the minimum distance between samples from class k and j
    # n is the dimensionality of sample space
    num_classes = len(r_set.keys())
    r_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        r_matrix[i, i] = r_set[i]

    # sort classes into priority groups with decreasing r
    priority_groups = defaultdict(list)
    for key, val in sorted(r_set.items()):
        priority_groups[val].append(key)

    # make a sorted list of values
    p_list = []
    for item in sorted(priority_groups.items()):
        key, val = item
        p_list.append([key, val])
    p_list = sorted(p_list)
    done_classes = []
    density_of_done = 0.0
    for k in range(len(p_list)):
        (r, curr_classes) = p_list[k]
        done_classes += curr_classes
        for c in curr_classes:
            # all classes in current priority group should have identical r value
            density_of_done += (1.0 / (r ** n))
        for i in curr_classes:
            for j in done_classes:
                # for each class you're looking at, we iterate through all
                # covered classes to add to the matrix
                if i != j:
                    r_matrix[i, j] = r_matrix[j, i] = 1.0 / (density_of_done ** (1.0 / n))  # r is symmetric
    return r_matrix


def find_most_underfilled_class(sample_count):
    return min(sample_count, key=sample_count.get)


def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def fill_rate(c, sample_count, num_target_samples):
    return sample_count[c] / num_target_samples[c]


def removable(conflicts, s, c, r_matrix):
    for past_pt, past_c in conflicts.items():
        if r_matrix[past_c, past_c] >= r_matrix[c, c] or fill_rate(past_c) < fill_rate(c):
            return False
    return True


def fill_grid(samples, r_set):
    '''
    Fill 2D image with Poisson sampled distribution of each class
    Intialize all points as the upper bound to find the pixels that could not be filled
    @param samples:
    @param r_set:
    @return:
    '''
    num_classes = len(r_set.keys())
    upper_bound = num_classes + 2
    grid = np.ones((width, height), dtype=np.float32) * upper_bound
    for pt, c in samples.items():
        grid[pt] = c

    holes = np.argwhere(grid == upper_bound)

    for i in range(holes.shape[0]):
        grid[holes[i, 0,], holes[i, 1]] = i % total_classes
    return grid


def multiclass_poisson(r_set, width):
    '''
    Uses multi-class poisson disk sampling method
    @param grid: sampling domain
    @param r_set: user specified params for intra-class sample spacing
    @return:
    '''
    num_classes = len(r_set.keys())
    r_matrix = build_r_matrix(r_set, n=2)
    samples = {}  # [coord: class]
    sample_count = dict.fromkeys(range(num_classes), 0)  # [class num: num samples for that class]
    max_trials = 2000000
    total_num_samples = width ** 2

    current_num_samples = 0  # count the number of sampples of all classes currently

    t = 0
    past_num = 0
    repeat_count = 0
    while t < max_trials and current_num_samples < total_num_samples:
        if t % 1000 == 0:
            if past_num == current_num_samples:
                repeat_count += 1
            else:
                repeat_count = 0
            past_num = current_num_samples
            print(f'Trials: {t}, num_samples: {current_num_samples}')
            if repeat_count == 10:
                break
        pt = (np.random.randint(0, width), np.random.randint(0, height))
        t += 1
        c = find_most_underfilled_class(sample_count)
        can_add = True

        conflicts = {}

        for key, val in samples.items():
            past_pt = key
            if distance(pt, past_pt) < r_matrix[c, val]:
                can_add = False
                conflicts[past_pt] = val
        if can_add:
            current_num_samples += 1
            sample_count[c] = sample_count[c] + 1
            samples[pt] = c
        else:  # impossible to add another sample to class c
            # try to remove the set of conflicting samples
            if removable(conflicts, pt, c, r_matrix):
                sample_count[c] = sample_count[c] + 1
                samples[pt] = c
                for past_pt, past_c in conflicts.items():
                    samples.pop(past_pt)
                    sample_count[past_c] = sample_count[past_c] - 1

    print(current_num_samples)
    print(sample_count)

    return fill_grid(samples), current_num_samples, sample_count


if __name__ == '__main__':
    width = height = 16
    [y, x] = np.mgrid[0:height, 0:width].astype(np.float64)

    total_classes = 8
    radius = 2  # use 2px radius for 8 classes
    # r_set must start at 0 and increase by 1 with each class
    # e.g. {0:2, 1:2, ..., 8:2} for 8 classes
    r_set = dict.fromkeys(range(total_classes), radius)

    # as determined by Eq(1)
    num_target_samples = dict.fromkeys(range(total_classes), int((width ** 2) / total_classes + 1))
    grid, current_num_samples, sample_count = multiclass_poisson(r_set, width=width)

    name = f'grid_16x16'

    np.save(f'{name}.npy', grid)
    torch.save(torch.tensor(grid), f'{name}.pt')

    file = open(f'output_{name}.txt', 'w')
    file.write(f'Sample Count: {current_num_samples}/{width ** 2}, {json.dumps(sample_count)}')
    file.close()

    plt.figure()
    plt.imshow(grid)
    plt.show()
