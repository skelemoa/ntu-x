import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 67
self_link = [(i, i) for i in range(num_node)]

# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

'''
# # VIBE 25 coordinates list : It follows openpose ordering
inward_ori_index = [(1, 2),  (3, 2), (4, 3), (5, 4), (6, 2),
                    (7, 6), (8, 7), (9, 2), (10, 9), (11, 10),
                    (12, 11), (13, 9), (14, 13), (15, 14), (16, 1),
                    (17, 1), (18, 16), (19, 17), (20, 15), (21, 20),
                    (22, 15), (23, 12), (24, 23), (25, 12)]
'''

# SMPL-X 67 joint ordering [Body - 25 joints, Left hand - 21 joints, Right hand - 21 joints]
inward_ori_index = [(1, 2), (3, 2), (4, 3), (5, 4), (6, 2), (7, 6),
                    (8, 7), (9, 2), (10, 9), (11, 10), (12, 11), (13, 9),
                    (14, 13), (15, 14), (16, 1), (17, 1), (18, 16), (19, 17),
                    (20, 15), (21, 20), (22, 15), (23, 12), (24, 23), (25, 12),
                    (26, 8), (27, 26), (28, 27), (29, 28), (30, 29), (31, 26),  # Left hand coordinates
                    (32, 31), (33, 32), (34, 33), (35, 26), (36, 35), (37, 36),
                    (38, 37), (39, 26), (40, 39), (41, 40), (42, 41), (43, 26),
                    (44, 43), (45, 44), (46, 45), (47, 5), (48, 47), (49, 48), # Right hand coordinates
                    (50, 49), (51, 50), (52, 47), (53, 52), (54, 53), (55, 54),
                    (56, 47), (57, 56), (58, 57), (59, 58), (60, 47), (61, 60),
                    (62, 61), (63, 62), (64, 47), (65, 64), (66, 65), (67, 66)]

                    
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
