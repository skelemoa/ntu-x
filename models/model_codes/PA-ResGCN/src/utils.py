import os, sys, shutil, logging, json, torch
from time import time, strftime, localtime


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def set_logging(save_dir):
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    handler = logging.FileHandler('{}/log.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def get_time(total_time):
    s = int(total_time % 60)
    m = int(total_time / 60) % 60
    h = int(total_time / 60 / 60) % 24
    d = int(total_time / 60 / 60 / 24)
    return '{:0>2d}d-{:0>2d}h-{:0>2d}m-{:0>2d}s'.format(d, h, m, s)


def get_current_timestamp():
    ct = time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)


def load_checkpoint(work_dir, model_name='resume'):
    if model_name == 'resume':
        file_name = '{}/checkpoint.pth.tar'.format(work_dir)
    elif model_name == 'debug':
        file_name = '{}/temp/debug.pth.tar'.format(work_dir)
    else:
        dirs, accs = {}, {}
        work_dir = '{}/{}'.format(work_dir, model_name)
        if os.path.exists(work_dir):
            for i, dir_time in enumerate(os.listdir(work_dir)):
                if os.path.isdir('{}/{}'.format(work_dir, dir_time)):
                    state_file = '{}/{}/reco_results.json'.format(work_dir, dir_time)
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            best_state = json.load(f)
                        accs[str(i+1)] = best_state['acc_top1']
                        dirs[str(i+1)] = dir_time
        if len(dirs) == 0:
            logging.warning('Warning: Do NOT exists any model in workdir!')
            logging.info('Evaluating initial or pretrained model.')
            return None
        logging.info('Please choose the evaluating model from the following models.')
        logging.info('Default is the initial or pretrained model.')
        for key in dirs.keys():
            logging.info('({}) accuracy: {:.2%} | training time: {}'.format(key, accs[key], dirs[key]))
        logging.info('Your choice (number of the model, q for quit): ')
        while True:
            idx = input(get_current_timestamp())
            if idx == '':
                logging.info('Evaluating initial or pretrained model.')
                return None
            elif idx in dirs.keys():
                break
            elif idx == 'q':
                logging.info('Quit!')
                sys.exit(1)
            else:
                logging.info('Wrong choice!')
        file_name = '{}/{}/{}.pth.tar'.format(work_dir, dirs[idx], model_name)
    if os.path.exists(file_name):
        return torch.load(file_name, map_location=torch.device('cpu'))
    else:
        logging.info('')
        logging.error('Error: Do NOT exist this checkpoint: {}!'.format(file_name))
        raise ValueError()


def save_checkpoint(model, optimizer, scheduler, epoch, best_state, is_best, work_dir, save_dir, model_name):
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {
        'model':model, 'optimizer':optimizer, 'scheduler':scheduler,
        'best_state':best_state, 'epoch':epoch,
    }
    cp_name = '{}/checkpoint.pth.tar'.format(work_dir)
    torch.save(checkpoint, cp_name)
    if is_best:
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_dir, model_name))
        with open('{}/reco_results.json'.format(save_dir), 'w') as f:
            del best_state['cm']
            json.dump(best_state, f)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
