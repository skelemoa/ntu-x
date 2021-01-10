import logging

from .ntu_generator import NTU_Generator
from .kinetics_generator import Kinetics_Generator
from .cmu_generator import CMU_Generator


__generator = {
    'ntu': NTU_Generator,
    'kinetics': Kinetics_Generator,
    'cmu': CMU_Generator,
}

def create(args):
    dataset = args.dataset.split('-')[0]
    dataset_args = args.dataset_args[dataset]
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(args.dataset))
        raise ValueError()
    return __generator[dataset](args, dataset_args)
