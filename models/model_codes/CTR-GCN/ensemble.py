import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            # npz_data = np.load('./data/' + 'ntu' + 'NTU_CS.npz')
            label_path = '/ssd_scratch/cvit/anirudh.thatipelli/ntu60x_data/xsub/test_label.pkl'
            with open(label_path, 'rb') as f:
                # self.sample_name, self.label = pickle.load(f, encoding='latin1')
                pickle_file = pickle.load(f)
                label = pickle_file[0][1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu' + 'NTU_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError


    final_dict = {}
    
    def addKeyToDict(key, data):
        data = data.reshape(1,-1)
        if key in final_dict.keys():
            final_dict[key] = np.concatenate([final_dict[key], data], axis=0)
        else:
            final_dict[key] = data

    with open(os.path.join(arg.joint_dir, 'epoch105_test_score.pkl'), 'rb') as r1:
        r1 = pickle.load(r1)
        for k in r1.keys():
            addKeyToDict(k, r1[k])

    with open(os.path.join(arg.bone_dir, 'epoch125_test_score.pkl'), 'rb') as r2:
        r2 = pickle.load(r2)
        for k in r2.keys():
            addKeyToDict(k, r2[k])

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch86_test_score.pkl'), 'rb') as r3:
            r3 = pickle.load(r3)
            for k in r3.keys():
                addKeyToDict(k, r3[k])
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch113_test_score.pkl'), 'rb') as r4:
            r4 = pickle.load(r4)
            for k in r4.keys():
                addKeyToDict(k, r4[k])

    right_num = total_num = right_num_5 = 0

    conf_mat = np.zeros([60,60])

    arg.alpha = [1, 1, 1, 1]
    for i in tqdm(range(len(final_dict.keys()))):
        k = list(final_dict.keys())[i]
        l = int(k[k.index("A") + 1 : k.index("A") + 4]) - 1 
        r11 = final_dict[k][0]
        r22 = final_dict[k][1]
        r33 = final_dict[k][2]
        r44 = final_dict[k][3]
        r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
        # r = r11 * arg.alpha[0] + r22 * arg.alpha[1]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        conf_mat[int(l),r] += 1
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    with open("conf_mat.npy", 'wb') as f:
        np.save(f, conf_mat)
    # if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
    #     arg.alpha = [0.6, 0.6, 0.4, 0.4]
    #     for i in tqdm(range(len(label))):
    #         l = label[i]
    #         _, r11 = r1[i]
    #         _, r22 = r2[i]
    #         _, r33 = r3[i]
    #         _, r44 = r4[i]
    #         r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
    #         rank_5 = r.argsort()[-5:]
    #         right_num_5 += int(int(l) in rank_5)
    #         r = np.argmax(r)
    #         right_num += int(r == int(l))
    #         total_num += 1
    #     acc = right_num / total_num
    #     acc5 = right_num_5 / total_num
    # elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
    #     arg.alpha = [0.6, 0.6, 0.4]
    #     for i in tqdm(range(len(label))):
    #         l = label[:, i]
    #         _, r11 = r1[i]
    #         _, r22 = r2[i]
    #         _, r33 = r3[i]
    #         r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
    #         rank_5 = r.argsort()[-5:]
    #         right_num_5 += int(int(l) in rank_5)
    #         r = np.argmax(r)
    #         right_num += int(r == int(l))
    #         total_num += 1
    #     acc = right_num / total_num
    #     acc5 = right_num_5 / total_num
    # else:
    #     for i in tqdm(range(len(label))):
    #         l = label[i]
    #         _, r11 = r1[i]
    #         _, r22 = r2[i]
    #         r = r11 + r22 * arg.alpha
    #         rank_5 = r.argsort()[-5:]
    #         right_num_5 += int(int(l) in rank_5)
    #         r = np.argmax(r)
    #         right_num += int(r == int(l))
    #         total_num += 1
    #     acc = right_num / total_num
    #     acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
