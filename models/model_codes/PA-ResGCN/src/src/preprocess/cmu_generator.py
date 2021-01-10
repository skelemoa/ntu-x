import os, pickle, logging, numpy as np

from .. import utils as U


class CMU_Generator():
    def __init__(self, args, dataset_args):
        self.in_path = dataset_args['cmu_data_path']
        self.out_path = '{}/{}'.format(dataset_args['path'], args.dataset)
        self.actions = ['walking', 'running', 'directing_traffic', 'soccer',
                        'basketball', 'washwindow', 'jumping', 'basketball_signal']
        self.dim_ignore = [0,  1,  2,  3,  4,  5,   6,   7,   8,  21,  22,  23, 24, 25, 26,
                          39, 40, 41, 60, 61, 62,  63,  64,  65,  81,  82,  83,
                          87, 88, 89, 90, 91, 92, 108, 109, 110, 114, 115, 116]
        self.dim_use = list(set(range(39*3)).difference(set(self.dim_ignore)))
        U.create_folder(self.out_path)

    def start(self):
        logging.info('Reading data ...')
        self.all_train_data, train_data = self.read_data('train')
        _, eval_data = self.read_data('test')

        logging.info('Normalizing data ...')
        self.data_mean, self.data_std, self.dim_zero, self.dim_nonzero = self.normalize_state()
        train_data = self.normalize_data(train_data)
        eval_data = self.normalize_data(eval_data)

        logging.info('Saving data ...')
        with open('{}/data.pkl'.format(self.out_path), 'wb') as f:
            pickle.dump((train_data, eval_data, self.actions), f)
        with open('{}/normalization.pkl'.format(self.out_path), 'wb') as f:
            pickle.dump((self.data_mean, self.data_std, self.dim_zero, self.dim_nonzero), f)
        with open('{}/ignore.pkl'.format(self.out_path), 'wb') as f:
            pickle.dump((self.dim_use, self.dim_ignore), f)

    def read_data(self, phase):
        all_data, even_data = [], {}
        for action_idx, action in enumerate(self.actions):
            action_path = '{}/{}/{}'.format(self.in_path, phase, action)
            for sequence_idx, file in enumerate(os.listdir(action_path)):
                sequence = []
                with open('{}/{}'.format(action_path, file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip().split(',')
                        if len(line) > 0:
                            sequence.append(np.array([np.float32(x) for x in line]))
                sequence = np.array(sequence)
                all_data.append(sequence)
                even_data[(action_idx, sequence_idx)] = sequence[range(0,sequence.shape[0],2),:]
        return np.concatenate(all_data, axis=0), even_data

    def normalize_state(self):
        data_mean = np.mean(self.all_train_data, axis=0)
        data_std = np.std(self.all_train_data, axis=0)
        dim_zero = list(np.where(data_std < 0.0001)[0])
        dim_nonzero = list(np.where(data_std >= 0.0001)[0])
        data_std[dim_zero] = 1.0
        return data_mean, data_std, dim_zero, dim_nonzero

    def normalize_data(self, data):
        for key in data.keys():
            data[key] = np.divide((data[key] - self.data_mean), self.data_std)
            data[key] = data[key][:, self.dim_use]
        return data
