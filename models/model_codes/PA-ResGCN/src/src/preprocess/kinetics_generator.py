import pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .kinetics_feeder import Kinetics_Feeder


# Thanks to SHI Lei for the released code on Github (https://github.com/lshiwjx/2s-AGCN)
class Kinetics_Generator():
    def __init__(self, args, dataset_args):
        self.num_joint = 18
        self.max_frame = 300
        self.num_person_out = 2
        self.num_person_in = 5
        self.print_bar = not args.no_progress_bar
        self.generate_label = args.generate_label

        self.in_path = dataset_args['kinetics_data_path']
        self.out_path = '{}/{}'.format(dataset_args['path'], args.dataset)
        U.create_folder(self.out_path)


    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase : {}'.format(phase))
            feeder = Kinetics_Feeder(
                data_path = '{}/kinetics_{}'.format(self.in_path, phase),
                label_path = '{}/kinetics_{}_label.json'.format(self.in_path, phase),
                max_frame = self.max_frame,
                num_person_in = self.num_person_in,
                num_person_out = self.num_person_out,
                num_joint = self.num_joint
            )
            sample_name = feeder.sample_name

            sample_label = []
            for i, s in enumerate(sample_name):
                sample_label.append(feeder[i][1])

            with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
                pickle.dump((sample_name, list(sample_label)), f)

            if not self.generate_label:
                fp = np.zeros((len(sample_name), 3, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)
                items = tqdm(sample_name, dynamic_ncols=True) if self.print_bar else sample_name
                for i, s in enumerate(items):
                    data = feeder[i][0]
                    fp[i, :, 0:data.shape[1], :, :] = data

                np.save('{}/{}_data.npy'.format(self.out_path, phase), fp)
