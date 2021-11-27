#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

num_nodes = 31
epsilon = 1e-6
bone_id = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                    [13, 17], [17, 18], [18, 19], [19, 20], [20, 21], [20, 23], [21, 22],
                    [13, 24], [24, 25], [25, 26], [26, 27], [27, 28], [27, 30], [28, 29]
                    ])
parent_joint = bone_id[:, 0]
son_joint = bone_id[:, 1]


def get_cmu_adjacent_matrix(debug=False, normalize=True):
    adjacent_matrix = np.zeros([num_nodes, num_nodes])
    for i in range(len(parent_joint)):
        adjacent_matrix[parent_joint[i], son_joint[i]] = 1
        adjacent_matrix[son_joint[i], parent_joint[i]] = 1
    if debug:
        np.savetxt('adj_matrix2.txt', adjacent_matrix, delimiter=',')
        plt.imshow(adjacent_matrix)
        plt.show()
        print(adjacent_matrix)
    if normalize:
        adjacent_matrix = normalize_incidence_matrix(adjacent_matrix)
    return adjacent_matrix


parent_joint_h36m = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
son_joint_h36m = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])


def get_h36m_adjacent_matrix():
    adjacent_matrix = np.zeros([17, 17])
    for i in range(len(parent_joint_h36m)):
        adjacent_matrix[parent_joint_h36m[i], son_joint_h36m[i]] = 1
        adjacent_matrix[son_joint_h36m[i], parent_joint_h36m[i]] = 1
        adjacent_matrix[i, i] = 1
    for i in range(len(adjacent_matrix[:, :])):
        adjacent_matrix[i, i] = 1
    print(adjacent_matrix)

    return adjacent_matrix


def normalize_incidence_matrix(im):
    im /= (im.sum(-1) + epsilon)[:, np.newaxis]
    return im


if __name__ == '__main__':
    get_h36m_adjacent_matrix()
