import os, pickle, logging, numpy as np
from torch.utils.data import Dataset

from .data_utils import *


class CMU_Feeder(Dataset):
    def __init__(self, phase, path, data_shape, connect_joint, repeat, debug, eval_batch_size, **kwargs):
        data_path = '{}/data.pkl'.format(path)
        normalize_path = '{}/normalization.pkl'.format(path)
        ignore_path = '{}/ignore.pkl'.format(path)
        if os.path.exists(data_path) and os.path.exists(normalize_path) and os.path.exists(ignore_path):
            with open(data_path, 'rb') as f:
                train_data, eval_data, self.actions = pickle.load(f, encoding='latin1')
            with open(normalize_path, 'rb') as f:
                self.data_mean, self.data_std, self.dim_zero, self.dim_nonzero = pickle.load(f, encoding='latin1')
            with open(ignore_path, 'rb') as f:
                self.dim_use, self.dim_ignore = pickle.load(f, encoding='latin1')
        else:
            train_data, eval_data = {}, {}
            logging.info('')
            logging.error('Error: Do NOT exist enough data files in {}! (3 files required)'.format(path))
            logging.info('Please generate data first!')
            raise ValueError()

        self.rng = np.random.RandomState(1234567890)
        self.conn = connect_joint
        self.T = data_shape[2]
        if phase == 'train':
            self.data = train_data
            self.select = np.random.randint(len(self.data), size=repeat*len(self.data))
        elif phase == 'eval':
            self.data = eval_data
            self.select = []
            for i in range(len(self.actions)):
                self.select += [i] * eval_batch_size
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this phase: {}'.format(phase))
            raise ValueError()
        if debug:
            self.select = self.select[:300]

    def __len__(self):
        return len(self.select)

    def __getitem__(self, idx):
        action_idx, sequence_idx = list(self.data)[self.select[idx]]
        name = '{}_{}'.format(self.actions[action_idx], sequence_idx)
        data = self.data[(action_idx, sequence_idx)]

        # (max_T, C*V*M) -> (T, C*V*M)
        max_T, d = data.shape
        frame_select = self.rng.randint(0, max_T-self.T)
        data = data[frame_select:frame_select+self.T, :]

        # (T, C*V*M) -> (C, T, V, M) -> (I, C*2, T, V, M)
        data = data.reshape(self.T, d//3, 3, 1).transpose(2, 0, 1, 3)
        data = multi_input(data, self.conn)

        return data, action_idx, name
