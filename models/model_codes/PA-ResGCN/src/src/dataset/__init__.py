import logging

from .graph import Graph
from .preprocessed import Preprocess_Feeder
from .ntu import NTU_Feeder, NTU_Location_Feeder
from .cmu import CMU_Feeder


__feeder = {
    'ntu-xsub': NTU_Feeder,
    'ntu-xview': NTU_Feeder,
    'ntu-xsub120': NTU_Feeder,
    'ntu-xset120': NTU_Feeder,
    'ntu_ax-xsub60': NTU_Feeder,
    'ntu_ax-xset60': NTU_Feeder,
    'ntu_hx-xsub60': NTU_Feeder,
    'ntu_hx-xset60': NTU_Feeder,
    'ntu-preprocess': Preprocess_Feeder,
    'kinetics': Preprocess_Feeder,
    'cmu': CMU_Feeder,
}

__shape = {
    'ntu-xsub': [3,6,300,25,2],
    'ntu-xview': [3,6,300,25,2],
    'ntu-xsub120': [3,6,300,25,2],
    'ntu-xset120': [3,6,300,25,2],
    'ntu_ax-xsub60': [3,6,300,118,2],
    'ntu_ax-xset60': [3,6,300,118,2],
    'ntu_hx-xsub60': [3,6,300,67,2],
    'ntu_hx-xset60': [3,6,300,67,2],
    'ntu-preprocess': [3,3,300,25,2],
    'kinetics': [3,3,300,18,2],
    'cmu': [3,6,50,26,1],
}

__class = {
    'ntu-xsub': 60,
    'ntu-xview': 60,
    'ntu_ax-xsub60': 60,
    'ntu_ax-xset60': 60,
    'ntu_hx-xsub60': 60,
    'ntu_hx-xset60': 60,
    'ntu-xsub120': 120,
    'ntu-xset120': 120,
    'kinetics': 400,
    'cmu': 8,
}

def create(debug, dataset, path, preprocess=False, **kwargs):
    if dataset not in __class.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.foramt(dataset))
        raise ValueError()
    graph = Graph(dataset)
    feeder_name = 'ntu-preprocess' if 'ntu' in dataset and preprocess else dataset
    kwargs.update({
        'path': '{}/{}'.format(path, dataset.replace('-', '/')),
        'data_shape': __shape[feeder_name],
        'connect_joint': graph.connect_joint,
        'debug': debug,
    })
    feeders = {
        'train': __feeder[feeder_name]('train', **kwargs),
        'eval' : __feeder[feeder_name]('eval', **kwargs),
    }
    if 'ntu' in dataset:
        feeders['ntu_location'] = NTU_Location_Feeder(__shape[feeder_name])
    return feeders, __shape[feeder_name], __class[dataset], graph.A, graph.parts
