import logging

from . import preprocess


class Generator():
    def __init__(self, args):
        self.dataset = args.dataset
        self.generator = preprocess.create(args)

    def start(self):
        logging.info('')
        logging.info('Starting generating ...')
        logging.info('Dataset: {}'.format(self.dataset))
        self.generator.start()
        logging.info('Finish generating!')
