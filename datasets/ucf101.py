import os
import random

from .video_dataset import VideoDataset
from ._paths import *


class UCF101(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, **kwargs):
        data = {}

        with open(split_file, 'r') as txt:
            data_path = [line.split()[0] for line in txt]
            data['video'] = [os.path.join(data_dir, p) for p in data_path]
            data['vid'] = data_path
        with open(vocab_file, 'r') as txt:
            class_name = [line.split()[1] for line in txt]
            data['label'] = [class_name.index(p.split('/')[0]) for p in data_path]
        
        super().__init__(data, **kwargs)


class UCF101FS(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, classes, **kwargs):
        data = {}

        with open(split_file, 'r') as txt:
            data_path = [line.split()[0] for line in txt]
        with open(vocab_file, 'r') as txt:
            class_name = [line.split()[1] for line in txt]
            
        data['video'] = []
        data['label'] = []
        for p in data_path:
            cls_id = class_name.index(p.split('/')[0])
            if cls_id in classes:
                data['video'].append(os.path.join(data_dir, p))
                data['label'].append(classes.index(cls_id))
        
        super().__init__(data, **kwargs)


def ucf101(data_dir=find_path(UCF_DATA),
           info_dir=find_path(UCF_META),
           train=True, split='01', **kwargs):

    phase = 'train' if train else 'test'
    info_list = os.path.join(info_dir, '{}list{}.txt'.format(phase, split))
    vocab_list = os.path.join(info_dir, 'classInd.txt')
    sampling = 'random' if train else 'uniform'
    return UCF101(data_dir, info_list, vocab_list, sampling=sampling, **kwargs)


def ucf101_fs(data_dir=find_path(UCF_DATA),
              info_dir=find_path(UCF_META),
              novel=False, split='01', phase='test', base_cls=0, **kwargs):

    n_cls = 101
    if base_cls > 0:
        subclass_fn = 'datasets/subclass/ucf/{:d}.txt'.format(base_cls)
        if os.path.isfile(subclass_fn):
            with open(subclass_fn, 'r') as f:
                subclass_list = [int(line.rstrip('\n')) for line in f]
        else:
            subclass_list = sorted(random.sample(range(n_cls), k=base_cls))
            os.makedirs(os.path.dirname(subclass_fn), exist_ok=True)
            with open(subclass_fn, 'w') as f:
                for v in subclass_list:
                    f.write(str(v) + '\n')
    else:
        subclass_list = []
    if novel:
        subclass_list = [k for k in range(n_cls) if k not in subclass_list]

    sampling = 'uniform' if novel else 'random'
    info_list = os.path.join(info_dir, '{}list{}.txt'.format(phase, split))
    vocab_list = os.path.join(info_dir, 'classInd.txt')
    return UCF101FS(data_dir, info_list, vocab_list, classes=subclass_list, sampling=sampling, **kwargs)