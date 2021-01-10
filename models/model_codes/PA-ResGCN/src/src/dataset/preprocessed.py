import os, pickle, logging, numpy as np
from torch.utils.data import Dataset

from .. import utils as U


class Preprocess_Feeder(Dataset):
    def __init__(self, phase, path, connect_joint, debug, **kwargs):
        self.conn = connect_joint
        data_path = '{}/{}_data.npy'.format(path, phase)
        label_path = '{}/{}_label.pkl'.format(path, phase)
        if os.path.exists(data_path) and os.path.exists(label_path):
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.sample_name = self.sample_name[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.sample_name[idx]

        # (C, T, V, M) -> (I, C, T, V, M)
        data = self.multi_input(data)

        return data, label, name

    def multi_input(self, data):
        C, T, V, M = data.shape
        data_new = np.zeros((3, C, T, V, M))
        data_new[0,:,:,:,:] = data
        for i in range(len(self.conn)):
            data_new[1,:,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        for i in range(T-1):
            data_new[2,:,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        data_new[2,:,T-1,:,:] = 0
        return data_new
