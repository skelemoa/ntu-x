import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .preprocessor import pre_normalization


# Thanks to SHI Lei for the released code on Github (https://github.com/lshiwjx/2s-AGCN)
class NTU_Generator():
    def __init__(self, args, dataset_args):
        self.num_person_out = 2
        self.num_person_in = 4
        self.num_joint = 25
        self.max_frame = 300
        self.dataset = args.dataset
        self.print_bar = not args.no_progress_bar
        self.generate_label = args.generate_label
        ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(__file__))

        self.out_path = '{}/{}'.format(dataset_args['path'], self.dataset.replace('-', '/'))
        U.create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['ntu-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))
        self.training_sample = training_samples[self.dataset]

        # Get ignore samples
        if os.path.exists(ntu_ignored):
            with open(ntu_ignored, 'r') as f:
                self.ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        else:
            logging.info('')
            logging.error('Error: Do NOT exist ignored sample file {}'.format(ntu_ignored))
            raise ValueError()

        # Get skeleton file list
        self.file_list = []
        for folder in [dataset_args['ntu60_data_path'], dataset_args['ntu120_data_path']]:
            for filename in os.listdir(folder):
                self.file_list.append((folder, filename))
            if '120' not in self.dataset:  # for NTU 60, only one folder
                break


    def start(self):
        # Generate data
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)


    def read_skeleton_filter(self, file):
        with open(file, 'r') as f:
            skeleton_sequence = {}
            skeleton_sequence['numFrame'] = int(f.readline())
            skeleton_sequence['frameInfo'] = []
            for t in range(skeleton_sequence['numFrame']):
                frame_info = {}
                frame_info['numBody'] = int(f.readline())
                frame_info['bodyInfo'] = []

                for m in range(frame_info['numBody']):
                    body_info = {}
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, f.readline().split())
                    }
                    body_info['numJoint'] = int(f.readline())
                    body_info['jointInfo'] = []
                    for v in range(body_info['numJoint']):
                        joint_info_key = [
                            'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                            'orientationW', 'orientationX', 'orientationY',
                            'orientationZ', 'trackingState'
                        ]
                        joint_info = {
                            k: float(v)
                            for k, v in zip(joint_info_key, f.readline().split())
                        }
                        body_info['jointInfo'].append(joint_info)
                    frame_info['bodyInfo'].append(body_info)
                skeleton_sequence['frameInfo'].append(frame_info)

        return skeleton_sequence


    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s


    def read_xyz(self, file):
        seq_info = self.read_skeleton_filter(file)
        data = np.zeros((self.num_person_in, seq_info['numFrame'], self.num_joint, 3))
        for n, f in enumerate(seq_info['frameInfo']):
            for m, b in enumerate(f['bodyInfo']):
                for j, v in enumerate(b['jointInfo']):
                    if m < self.num_person_in and j < self.num_joint:
                        data[m, n, j, :] = [v['x'], v['y'], v['z']]

        # select two max energy body
        energy = np.array([self.get_nonzero_std(x) for x in data])
        index = energy.argsort()[::-1][0:self.num_person_out]
        data = data[index]

        data = data.transpose(3, 1, 2, 0)  # to (C,T,V,M)
        return data


    def gendata(self, phase):
        sample_name = []
        sample_label = []
        sample_paths = []
        for folder, filename in sorted(self.file_list):
            if filename in self.ignored_samples:
                continue

            path = os.path.join(folder, filename)
            setup_loc = filename.find('S')
            camera_loc = filename.find('C')
            subject_loc = filename.find('P')
            action_loc = filename.find('A')
            setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
            camera_id = int(filename[(camera_loc+1):(camera_loc+4)])
            subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
            action_class = int(filename[(action_loc+1):(action_loc+4)])

            if self.dataset == 'ntu-xview':
                istraining = (camera_id in self.training_sample)
            elif self.dataset == 'ntu-xsub' or self.dataset == 'ntu-xsub120':
                istraining = (subject_id in self.training_sample)
            elif self.dataset == 'ntu-xset120':
                istraining = (setup_id in self.training_sample)
            else:
                istraining = None
                logging.info('')
                logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                raise ValueError()

            if phase == 'train':
                issample = istraining
            elif phase == 'eval':
                issample = not (istraining)
            else:
                issample = None
                logging.info('')
                logging.error('Error: Do NOT exist this phase {}'.format(phase))
                raise ValueError()

            if issample:
                sample_paths.append(path)
                sample_label.append(action_class - 1)  # to 0-indexed

        # Save labels
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_paths, list(sample_label)), f)

        if not self.generate_label:
            # Create data tensor (N,C,T,V,M)
            fp = np.zeros((len(sample_label), 3, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)

            # Fill (C,T,V,M) to data tensor (N,C,T,V,M)
            items = tqdm(sample_paths, dynamic_ncols=True) if self.print_bar else sample_paths
            for i, s in enumerate(items):
                data = self.read_xyz(s)
                fp[i, :, 0:data.shape[1], :, :] = data

            # Perform preprocessing on data tensor
            fp = pre_normalization(fp, print_bar=self.print_bar)

            # Save input data (train/eval)
            np.save('{}/{}_data.npy'.format(self.out_path, phase), fp)
