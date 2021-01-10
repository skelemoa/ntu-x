import logging, torch, numpy as np
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
import logging

writer = SummaryWriter("/home/anirudh.thatipelli/runs_resgcn_ntu60_smpl_joint_118_xsub/")
loss_writer = SummaryWriter("/home/anirudh.thatipelli/runs_resgcn_ntu60_smpl_joint_118_xsub/")

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level = logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# logger for training accuracies
train_logger = setup_logger('Training_accuracy', '/home/anirudh.thatipelli/runs_resgcn_ntu60_smpl_joint_118_xsub/train_output.log')

# logger for evaluation accuracies
eval_logger = setup_logger('Evaluation_accuracy', '/home/anirudh.thatipelli/runs_resgcn_ntu60_smpl_joint_118_xsub/eval_output.log')


from . import utils as U
from .initializer import Initializer


class Processor(Initializer):

    def calc_mad(self, acc_values):
      med_acc_values = torch.median(torch.stack(acc_values))

      abs_dev_med = []
      for ind in range(len(acc_values)):
        abs_dev_med.append(torch.abs(acc_values[ind] - med_acc_values))

      med_abs_dev = torch.median(torch.stack(abs_dev_med))

      return med_abs_dev

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        train_acc, num_sample = 0, 0
        loss_value = []
        acc_value = []
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            loss = self.loss_func(out, y)
            loss.backward()
            loss_value.append(loss)

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            train_acc += reco_top1.eq(y).sum().item()
          
            value, predict_label = torch.max(out.data, 1)
            acc = torch.mean((predict_label == y.data).float())
            acc_value.append(acc)

            
            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Saving the model
        # state_dict = self.model.state_dict()
        # weights = OrderedDict([[k.split('module.')[-1],
        #                         v.cpu()] for k, v in state_dict.items()])

        # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

        # Showing Train Results
        train_acc /= num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:.2%}, Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, train_acc, time()-start_train_time
        ))
        logging.info('')

        std_acc = torch.std(torch.stack(acc_value))
        med_abs_dev = self.calc_mad(acc_value)
        avg_loss = np.mean(loss_value)
        avg_acc = torch.mean(torch.stack(acc_value))
        med_loss = np.median(loss_value)
        med_acc = torch.median(torch.stack(acc_value))

        return avg_loss, med_loss, avg_acc, med_acc, std_acc, med_abs_dev


    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            acc_top1, acc_top5 = 0, 0
            num_sample, eval_loss = 0, []
            loss_value = []
            acc_value = []
            score_frag = []
            score_dict = {}
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, name) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                loss_value.append(loss.data.cpu().numpy())

                _, predict_label = torch.max(out.data, 1)
                acc = torch.mean((predict_label == y.data).float())
                acc_value.append(acc)

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                acc_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out,5)[1]
                acc_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Saving the output for each corresponding sample files
                score_frag.append(out.data.cpu().numpy())
                score_dict[name] = out.data.cpu().numpy()

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        score = np.concatenate(score_frag)

        # Saving the corresponding 
        std_acc = torch.std(torch.stack(acc_value))
        med_abs_dev = self.calc_mad(acc_value)
        avg_loss = np.mean(loss_value)
        avg_acc = torch.mean(torch.stack(acc_value))
        med_loss = np.median(loss_value)
        med_acc = torch.median(torch.stack(acc_value))

        with open('./work_dir/' + arg.Experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(avg_acc) + '.pkl'.format(
                epoch, accuracy), 'wb') as f:
            pickle.dump(score_dict, f)

        # Showing Evaluating Results
        acc_top1 /= num_sample
        acc_top5 /= num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:.2%}, Top-5 accuracy: {:.2%}, Mean loss:{:.4f}'.format(
            acc_top1, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        return avg_loss, med_loss, avg_acc, med_acc, std_acc, med_abs_dev

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                avg_train_loss, med_train_loss, avg_train_acc, med_train_acc = self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))

                avg_val_loss, med_val_loss, avg_val_acc, med_val_acc, std_dev_acc, med_abs_dev = self.eval()

                writer.add_scalars('Average Accuracies', {'Average training accuracy': avg_train_acc, 'Average Validation accuracy': avg_val_acc, 
                    'Median training accuracy': med_train_acc, "Median Validation accuracy" : med_val_acc }, epoch + 1)
                loss_writer.add_scalars('Average Losses', {'Average training loss': avg_train_loss, 'Average Validation loss': avg_val_loss, 
                    "Median training loss" : med_train_loss, "Median Validation loss": med_val_loss }, epoch + 1)

                train_logger.info("Average training loss after {0} epochs is {1}".format(epoch + 1, avg_train_loss))
                train_logger.info("Average training accuracy after {0} epochs is {1}".format(epoch + 1, avg_train_acc))
                train_logger.info("Median training loss after {0} epochs is {1}".format(epoch + 1, med_train_loss))
                train_logger.info("Median training accuracy after {0} epochs is {1}".format(epoch + 1, med_train_acc))

                eval_logger.info("Average validation loss after {0} epochs is {1}".format(epoch + 1, avg_val_loss))
                eval_logger.info("Average validation accuracy after {0} epochs is {1}".format(epoch + 1, avg_val_acc))
                eval_logger.info("Median validation loss after {0} epochs is {1}".format(epoch + 1, med_val_loss))
                eval_logger.info("Median validation accuracy after {0} epochs is {1}".format(epoch + 1, med_val_acc))
                eval_logger.info("Standard deviation of validation accuracy after {0} epochs is {1}".format(epoch + 1, std_dev_acc))
                eval_logger.info("Median absolute deviation of validation accuracy after {0} epochs is {1}".format(epoch + 1, med_abs_dev))

                    # acc_top1, acc_top5, cm = self.eval()
                    # if acc_top1 > best_state['acc_top1']:
                    #     is_best = True
                    #     best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        cm = checkpoint['best_state']['cm']
        self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.fcn.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=cm,
                feature=feature, weight=weight, location=location
            )
        logging.info('Finish extracting!')
        logging.info('')
