import os
import torch
import random

from .video_dataset import VideoDataset
from ._paths import *
from .utils.torch_videovision.torchvideotransforms import video_transforms, volume_transforms


class HMDB(VideoDataset):
    def __init__(self, data_dir, split_dir, to_read, split, **kwargs):
        data = {}
        data['video'] = []
        data['vid'] = []
        data['label'] = []
        vocab = []
        class_idx = 0
        for p in os.listdir(split_dir):
            if p.endswith('test_split{}.txt'.format(split)):
                class_name = p.rsplit('_', 2)[0]
                vocab.append(class_name)
                with open(os.path.join(split_dir, p), 'r') as txt:
                    for line in txt:
                        vid_name, tag = line.split()
                        if tag == str(to_read):
                            data['video'].append(os.path.join(data_dir, class_name, vid_name))
                            data['vid'].append(vid_name)
                            data['label'].append(class_idx)
                class_idx += 1

        super().__init__(data, **kwargs)
        self.vocab = vocab


class HMDBFS(VideoDataset):
    def __init__(self, data_dir, split_dir, classes, split, **kwargs):
        data = {}
        data['video'] = []
        data['label'] = []
        class_idx = 0
        for p in os.listdir(split_dir):
            if p.endswith('test_split{}.txt'.format(split)):
                class_name = p.rsplit('_', 2)[0]
                with open(os.path.join(split_dir, p), 'r') as txt:
                    for line in txt:
                        vid_name, tag = line.split()
                        if class_idx in classes:
                            data['video'].append(os.path.join(data_dir, class_name, vid_name))
                            data['label'].append(classes.index(class_idx))
                class_idx += 1

        super().__init__(data, **kwargs)


def hmdb(data_dir=find_path(HMDB_DATA),
         info_dir=find_path(HMDB_META),
         train=True, split=1, **kwargs):

    to_read = 1 if train else 2
    sampling = 'random' if train else 'uniform'
    return HMDB(data_dir, info_dir, to_read=to_read, split=split, sampling=sampling, **kwargs)


def hmdb_fs(data_dir=find_path(HMDB_DATA),
            info_dir=find_path(HMDB_META),
            novel=False, split=1, base_cls=0, **kwargs):

    n_cls = 51
    if base_cls > 0:
        subclass_fn = 'datasets/subclass/hmdb/{:d}.txt'.format(base_cls)
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
    return HMDBFS(data_dir, info_dir, classes=subclass_list, split=split, sampling=sampling, **kwargs)