import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import os
import h5py
import pickle

train_data_path =  "/ssd_scratch/cvit/anirudh.thatipelli/ntu60x_data/xsub/train/"
test_data_path = "/ssd_scratch/cvit/anirudh.thatipelli/ntu60x_data/xsub/test/"

train_subj_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 
93, 94, 95, 97, 98, 100, 103]

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        
        label_path = '/ssd_scratch/cvit/anirudh.thatipelli/ntu60x_data/xsub/{}_label.pkl'.format(self.split)
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                # self.sample_name, self.label = pickle.load(f, encoding='latin1')
                self.pickle_file = pickle.load(f)
                self.sample_name = self.pickle_file[0][0]
                self.label = self.pickle_file[0][1]
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {}!'.format(label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if self.debug:
            self.sample_name = self.sample_name[:300]
            self.label = self.label[:300]

        print(split + " data samples:" + str(len(self.sample_name)))
        # self.load_data()
        # if normalization:
            # self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, idx):

        label = self.label[idx]
        name = self.sample_name[idx].split(".")[0] + ".h5"

        subj_id = int(name[name.index("P") + 1 : name.index("P") + 4])
        if subj_id in train_subj_ids:
           # data = np.load(train_data_path + name)
           d = h5py.File(train_data_path + name,'r')
        else:
           # data = np.load(test_data_path + name)
           d = h5py.File(test_data_path + name,'r')
        
        data = d[list(d.keys())[-1]][:]
        data_numpy = data[:, :, :67, :] 


        '''data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)'''
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, idx

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
