import sys

sys.path.extend(['../'])
from graph import tools

num_node = 67
self_link = [(i, i) for i in range(num_node)]
'''
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
'''

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
'''
                     (68, 69), (69, 70), (70, 71), (71, 72), (73, 74), (74, 75), # Face coordinates
                     (75, 76), (76, 77), (78, 1), (79, 78), (80, 79), (81, 80),
                     (82, 83), (83, 84), (84, 85), (85, 86), (87, 16), (88, 87),
                     (89, 88), (90, 89), (91, 90), (92, 91), (92, 87), (93, 17),
                     (94, 93), (95, 94), (96, 95), (97, 96), (98, 97), (98, 93),
                     (99, 100), (100, 101), (101, 102), (102, 103), (103, 104),
                     (104, 105), (105, 106), (106, 107), (107, 108), (108, 109),
                     (109, 110), (110, 99), (111, 112), (112, 113), (113, 114),
                     (114, 115), (115, 116), (116, 117), (117, 118), (118, 111)]
'''

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
