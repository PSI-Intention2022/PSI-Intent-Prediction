import os
import pickle
import numpy as np
import torch
import json
# from create_database import create_intent_database
from data.process_sequence import generate_data_sequence
from data.custom_dataset import VideoDataset

def get_dataloader(args, shuffle_train=True, drop_last_train=True):
    with open(os.path.join(args.database_path, 'intent_database_train.pkl'), 'rb') as fid:
        imdb_train = pickle.load(fid)
    train_seq = generate_data_sequence('train', imdb_train, args)
    with open(os.path.join(args.database_path, 'intent_database_val.pkl'), 'rb') as fid:
        imdb_val = pickle.load(fid)
    val_seq = generate_data_sequence('val', imdb_val, args)
    with open(os.path.join(args.database_path, 'intent_database_test.pkl'), 'rb') as fid:
        imdb_test = pickle.load(fid)
    test_seq = generate_data_sequence('test', imdb_test, args)

    train_d = get_train_val_data(train_seq, args, overlap=args.seq_overlap_rate) # returned tracks
    val_d = get_train_val_data(val_seq, args, overlap=args.test_seq_overlap_rate)
    test_d = get_test_data(test_seq, args, overlap=args.test_seq_overlap_rate)

    # Create video dataset and dataloader
    train_dataset = VideoDataset(train_d, args)
    val_dataset = VideoDataset(val_d, args)
    test_dataset = VideoDataset(test_d, args)
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train,
                                           pin_memory=True, sampler=None, drop_last=drop_last_train, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              pin_memory=True, sampler=None, drop_last=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              pin_memory=True, sampler=None, drop_last=False, num_workers=4)

    return train_loader, val_loader, test_loader


def get_train_val_data(data, args, overlap=0.5):  # overlap==0.5, seq_len=15
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    print("Train/Val Tracks: ", tracks.keys())
    return tracks


def get_test_data(data, args, overlap=1):  # overlap==0.5, seq_len=15
    # return splited train/val dataset
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    print("Test Tracks: ", tracks.keys())
    return tracks


def get_tracks(data, seq_len, observed_seq_len, overlap, args):
    overlap_stride = observed_seq_len if overlap == 0 else \
        int((1 - overlap) * observed_seq_len)  # default: int(0.5*15) == 7

    overlap_stride = 1 if overlap_stride < 1 else overlap_stride # when test, overlap=1, stride=1

    d_types = ['video_id', 'ped_id', 'frame', 'bbox', 'intention_binary', 'intention_prob', 'disagree_score', 'description']

    d = {}

    for k in d_types:
        d[k] = data[k]

    for k in d.keys():
        # print(k, len(d[k]))
        # frame/bbox/intention_binary/reason_feats
        tracks = []
        for track_id in range(len(d[k])):
            track = d[k][track_id]
            ''' There are some sequences not adjacent '''
            frame_list = data['frame'][track_id]
            if len(frame_list) < args.max_track_size: #60:
                print('too few frames: ', d['video_id'][track_id][0], d['ped_id'][track_id][0])
                continue
            splits = []
            start = -1
            for fid in range(len(frame_list) - 1):
                if start == -1:
                    start = fid  # frame_list[f]
                if frame_list[fid] + 1 == frame_list[fid + 1]:
                    if fid + 1 == len(frame_list) - 1:
                        splits.append([start, fid + 1])
                    continue
                else:
                    # current f is the end of current piece
                    splits.append([start, fid])
                    start = -1
            if len(splits) != 1:
                print('NOT one missing split found: ', splits)
                raise Exception()
            else: # len(splits) == 1, No missing frames from the database.
                pass
            sub_tracks = []
            for spl in splits:
                # explain the boundary:  end_idx - (15-1=14 gap) + cover last idx
                for i in range(spl[0], spl[1] - (seq_len - 1) + 1, overlap_stride):
                    sub_tracks.append(track[i:i + seq_len])
            tracks.extend(sub_tracks)

        d[k] = np.array(tracks)
    return d
