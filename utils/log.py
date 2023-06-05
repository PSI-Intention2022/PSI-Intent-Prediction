import os
import numpy as np
from utils.utils import AverageMeter
from utils.metrics import evaluate_intent
import json


class RecordResults():
    def __init__(self, args=None, intent=True, traj=True, reason=False, evidential=False,
                 extract_prediction=False):
        self.args = args
        self.save_output = extract_prediction
        self.intent = intent
        self.traj = traj
        self.reason = reason
        self.evidential = evidential

        self.all_train_results = {}
        self.all_eval_results = {}
        self.all_val_results = {}

        # cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        self.result_path = os.path.join(self.args.checkpoint_path, 'results')
        if not os.path.isdir(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)

        self._log_file = os.path.join(self.args.checkpoint_path, 'log.txt')
        open(self._log_file, 'w').close()

        # self.log_args(self.args)

    def log_args(self, args):
        args_file = os.path.join(self.args.checkpoint_path, 'args.txt')
        with open(args_file, 'a') as f:
            json.dump(args.__dict__, f, indent=2)
        ''' 
            parser = ArgumentParser()
            args = parser.parse_args()
            with open('commandline_args.txt', 'r') as f:
            args.__dict__ = json.load(f)
        '''

    def train_epoch_reset(self, epoch, nitern):
        # 1. initialize log info
        # (1.1) loss log list
        self.log_loss_total = AverageMeter()
        self.log_loss_intent = AverageMeter()
        self.log_loss_traj = AverageMeter()
        # (1.2) intent
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        # (1.3) trajectory - args.image_shape
        self.traj_gt = []  # normalized, N x 4, (0, 1) range
        self.traj_pred = []  # N x 4 dimension, (0, 1) range
        # (1.4) store all results
        self.train_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern


    def train_intent_batch_update(self, itern, data, intent_gt, intent_prob_gt, intent_prob, loss, loss_intent):
        # 3. Update training info
        # (3.1) loss log list
        bs = intent_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_intent.update(loss_intent, bs)
        # (3.2) training data info
        if intent_prob != []:
            # (3.3) intent
            self.intention_gt.extend(intent_gt)  # bs
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)  # bs

            # assert len(self.intention_gt[0]) == 1 #self.args.predict_length, intent only predict 1 result
        else:
            pass

        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path+"/training_info.txt", 'a') as f:
                f.write('Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  Intent Loss: {:.4f} \n'.format(
                    self.epoch, self.args.epochs, itern, self.nitern, self.log_loss_total.avg,
                    self.log_loss_intent.avg))


    def train_intent_epoch_calculate(self, writer=None):
        print('----------- Training results: ------------------------------------ ')
        if self.intention_pred:
            intent_results = evaluate_intent(np.array(self.intention_gt), np.array(self.intention_prob_gt),
                                             np.array(self.intention_pred), self.args)
            self.train_epoch_results['intent_results'] = intent_results

        print('----------------------------------------------------------- ')
        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename='train')

        # write scalar to tensorboard
        if writer:
            for key in ['MSE', 'Acc', 'F1', 'mAcc']:
                val = intent_results[key]
                writer.add_scalar(f'Train/Results/{key}', val, self.epoch)

            for i in range(self.args.intent_num):
                for j in range(self.args.intent_num):
                    val = intent_results['ConfusionMatrix'][i][j]
                    writer.add_scalar(f'ConfusionMatrix/train{i}_{j}', val, self.epoch)

    def eval_epoch_reset(self, epoch, nitern, intent=True, traj=True, args=None):
        # 1. initialize log info
        # (1.2) training data info
        self.frames_list = []
        self.video_list = []
        self.ped_list = []
        # (1.3) intent
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        self.intention_rsn_gt = []
        self.intention_rsn_pred = []
        
        # (1.4) trajectory - args.image_shape
        self.traj_gt = []  # normalized, N x 4, (0, 1) range
        self.traj_pred = []  # N x 4 dimension, (0, 1) range

        self.eval_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern


    def eval_intent_batch_update(self, itern, data, intent_gt, intent_prob, intent_prob_gt, intent_rsn_gt=None, intent_rsn_pred=None):
        # 3. Update training info
        # (3.1) loss log list
        bs = intent_gt.shape[0]
        # (3.2) training data info
        self.frames_list.extend(data['frames'].detach().cpu().numpy())  # bs x sq_length(60)
        assert len(self.frames_list[0]) == self.args.observe_length
        self.video_list.extend(data['video_id'])  # bs
        self.ped_list.extend(data['ped_id'])
        # print("save record: video list - ", data['video_id'])

        # (3.3) intent
        if intent_prob != []:
            self.intention_gt.extend(intent_gt)  # bs
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)  # bs
            if intent_rsn_gt is not None:
                self.intention_rsn_gt.extend(intent_rsn_gt)
                self.intention_rsn_pred.extend(intent_rsn_pred)
            # assert len(self.intention_gt[0]) == 1 #self.args.predict_length, intent only predict 1 result
        else:
            pass

    def eval_intent_epoch_calculate(self, writer):
        print('----------- Evaluate results: ------------------------------------ ')

        if self.intention_pred:
            intent_results = evaluate_intent(np.array(self.intention_gt), np.array(self.intention_prob_gt),
                                             np.array(self.intention_pred), self.args)
            self.eval_epoch_results['intent_results'] = intent_results

        print('----------------------finished evalcal------------------------------------- ')
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename='eval')
        print('log info finished')

        # write scalar to tensorboard
        if writer:
            for key in ['MSE', 'Acc', 'F1', 'mAcc']:
                val = intent_results[key]
                writer.add_scalar(f'Eval/Results/{key}', val, self.epoch)

            for i in range(self.args.intent_num):
                for j in range(self.args.intent_num):
                    val = intent_results['ConfusionMatrix'][i][j]
                    writer.add_scalar(f'ConfusionMatrix/eval{i}_{j}', val, self.epoch)

    # def save_results(self, prefix=''):
    #     self.result_path = os.path.join(self.args.checkpoint_path, 'results', f'epoch_{self.epoch}', prefix)
    #     if not os.path.isdir(self.result_path):
    #         os.makedirs(self.result_path)
    #     # 1. train results
    #     np.save(self.result_path + "/train_results.npy", self.all_train_results)
    #     # 2. eval results
    #     np.save(self.result_path + "/eval_results.npy", self.all_eval_results)
    #
    #     # 3. save data
    #     np.save(self.result_path + "/intent_gt.npy", self.intention_gt)
    #     np.save(self.result_path + "/intent_prob_gt.npy", self.intention_prob_gt)
    #     np.save(self.result_path + "/intent_pred.npy", self.intention_pred)
    #     np.save(self.result_path + "/frames_list.npy", self.frames_list)
    #     np.save(self.result_path + "/video_list.npy", self.video_list)
    #     np.save(self.result_path + "/ped_list.npy", self.ped_list)
    #     np.save(self.result_path + "/intent_rsn_gt.npy", self.intention_rsn_gt)
    #     np.save(self.result_path + "/intent_rsn_pred.npy", self.intention_rsn_pred)
    #


    def log_msg(self, msg: str, filename: str = None):
        if not filename:
            filename = os.path.join(self.args.checkpoint_path, 'log.txt')
        else:
            pass
        savet_to_file = filename
        with open(savet_to_file, 'a') as f:
            f.write(str(msg) + '\n')

    def log_info(self, epoch: int, info: dict, filename: str = None):
        if not filename:
            filename = 'log.txt'
        else:
            pass
        for key in info:
            savet_to_file = os.path.join(self.args.checkpoint_path, filename + '_' + key + '.txt')
            self.log_msg(msg='Epoch {} \n --------------------------'.format(epoch), filename=savet_to_file)
            with open(savet_to_file, 'a') as f:
                    if type(info[key]) == str:
                        f.write(info[key] + "\n")
                    elif type(info[key]) == dict:
                        for k in info[key]:
                            f.write(k + ": " + str(info[key][k]) + "\n")
                    else:
                        f.write(str(info[key]) + "\n")
            self.log_msg(msg='.................................................'.format(self.epoch), filename=savet_to_file)


