import os, warnings, logging, pynvml, torch, numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import utils as U
from . import dataset
from . import model
from . import scheduler


class Initializer():
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = meminfo.used / 1024 / 1024
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                if memused > 1000:
                    pynvml.nvmlShutdown()
                    logging.info('')
                    logging.error('GPU-{} is occupied!'.format(i))
                    raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.debug, self.args.dataset, **dataset_args
        )
        self.train_loader = DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=True, drop_last=True
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=False, drop_last=False
        )
        self.location_loader = self.feeders['ntu_location'] if dataset_name == 'ntu' else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': torch.Tensor(self.A),
            'parts': [torch.Tensor(part).long() for part in self.parts]
        }
        self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs).to(self.device)
        self.model = torch.nn.DataParallel(
            self.model, device_ids=self.args.gpus, output_device=self.output_device
        )
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        logging.info('Model parameters: {:.2f}M'.format(
            sum(p.numel() for p in self.model.parameters()) / 1000 / 1000
        ))
        pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
            self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}'.format(pretrained_model))

    def init_optimizer(self):
        try:
            optimizer = U.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.info('Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))
