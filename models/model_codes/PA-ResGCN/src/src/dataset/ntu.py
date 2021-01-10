import os, pickle, logging, numpy as np
from torch.utils.data import Dataset

from .data_utils import *


class NTU_Feeder(Dataset):
    def __init__(self, phase, path, data_shape, connect_joint, debug, **kwargs):
        _, _, self.T, self.V, self.M = data_shape
        self.conn = connect_joint
        label_path = '{}/{}_label.pkl'.format(path, phase)
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {}!'.format(label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.sample_name = self.sample_name[:300]
            self.label = self.label[:300]

    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, idx):
        label = self.label[idx]
        name = self.sample_name[idx]

        data = np.zeros((3, self.T, self.V, self.M))
        with open(name, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                if frame >= self.T:
                    break
                person_num = int(fr.readline())
                for person in range(person_num):
                    fr.readline()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        v = fr.readline().split(' ')
                        if joint < self.V and person < self.M:
                            data[0,frame,joint,person] = float(v[0])
                            data[1,frame,joint,person] = float(v[1])
                            data[2,frame,joint,person] = float(v[2])

        # (C, T, V, M) -> (I, C*2, T, V, M)
        data = multi_input(data, self.conn)

        return data, label, name


class NTU_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i,0,frame,joint,person] = float(v[5])
                                location[i,1,frame,joint,person] = float(v[6])
        return location
