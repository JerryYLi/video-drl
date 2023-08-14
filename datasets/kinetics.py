import os
import csv
import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from .video_dataset import VideoDataset
from ._paths import *
from .utils.torch_videovision.torchvideotransforms import video_transforms, volume_transforms


class Kinetics(VideoDataset):
    def __init__(self, data_dir, split_file, vocab, subclass=0, exclude=False, dset_name='kinetics', phase='train', **kwargs):
        data = {}

        # labels
        if isinstance(vocab, str):
            with open(vocab, 'r') as txt:
                vocab = [line.rstrip() for line in txt]
        
        # use partial classes only
        if subclass > 0:
            n_class = len(vocab)
            subclass_fn = 'datasets/subclass/{:s}/{:d}.txt'.format(dset_name, subclass)
            if os.path.isfile(subclass_fn):
                with open(subclass_fn, 'r') as f:
                    subclass_list = [int(line.rstrip('\n')) for line in f]
            else:
                subclass_list = sorted(random.sample(range(n_class), k=subclass))
                os.makedirs(os.path.dirname(subclass_fn), exist_ok=True)
                with open(subclass_fn, 'w') as f:
                    for v in subclass_list:
                        f.write(str(v) + '\n')
            if exclude:
                subclass_list = [k for k in range(n_class) if k not in subclass_list]
        else:
            subclass_list = []
        
        # csv file
        data['video'] = []
        data['label'] = []
        data['vid'] = []
        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                count += 1
                if count % 10000 == 0:
                    print('{}/{} videos'.format(len(data['label']), count))
                vid_path = os.path.join(data_dir, row['label'], row['youtube_id'] + '_{:06d}_{:06d}.mp4'
                    .format(int(float(row['time_start'])), int(float(row['time_end']))))
                if os.path.isfile(vid_path):
                    if row['label'] in vocab:
                        class_id = vocab.index(row['label'])
                        if subclass_list:
                            if class_id in subclass_list:
                                class_id = subclass_list.index(class_id)
                            else:
                                continue
                        data['video'].append(vid_path)
                        data['label'].append(class_id)
                        data['vid'].append(os.path.basename(vid_path).replace('.mp4', ''))
            print('{}/{} videos'.format(len(data['label']), count))

        super().__init__(data, **kwargs)
        self.vocab = vocab


def kinetics_400(data_dir=find_path(K400_DATA),
                 info_dir=find_path(K400_META),
                 train=True, **kwargs):

    phase = 'train' if train else 'validate'
    split_dir = os.path.join(data_dir, phase)
    if os.path.isdir(split_dir):
        data_dir = split_dir
    info_list = os.path.join(info_dir, '{0}.csv'.format(phase))
    vocab_list = os.path.join(info_dir, 'label_map.txt')
    sampling = 'random' if train else 'uniform'
    return Kinetics(data_dir, info_list, vocab_list, dset_name='k400', phase=phase, sampling=sampling, **kwargs)


def mini_kinetics_200(data_dir=find_path(K400_DATA),
                      info_dir=find_path(K200_META),
                      train=True, **kwargs):

    phase = 'train' if train else 'validate'
    split_dir = os.path.join(data_dir, phase)
    if os.path.isdir(split_dir):
        data_dir = split_dir
    info_list = os.path.join(info_dir, '{0}.csv'.format(phase))
    vocab_list = os.path.join(info_dir, 'label_map.txt')
    sampling = 'random' if train else 'uniform'
    return Kinetics(data_dir, info_list, vocab_list, dset_name='minikinetics', phase=phase, sampling=sampling, **kwargs)