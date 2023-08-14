import os
import json
import torch
import random

from .video_dataset import VideoDataset
from ._paths import *
from .utils.torch_videovision.torchvideotransforms import video_transforms, volume_transforms


class Diving(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, classes=None, **kwargs):
        data = {}

        with open(split_file, 'r') as f:
            anno = json.load(f)
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
            
        data['video'] = []
        data['vid'] = []
        data['label'] = []
        for vid in anno:
            label_id = vid['label']
            if classes is not None:
                if label_id in classes:
                    label_id = classes.index(label_id)
                else:
                    continue
            data['video'].append(os.path.join(data_dir, vid['vid_name'] + '.mp4'))
            data['vid'].append(vid['vid_name'])
            data['label'].append(label_id)
        
        super().__init__(data, **kwargs)


def diving(data_dir=find_path(DIVING_DATA),
           info_dir=find_path(DIVING_META),
           train=True, **kwargs):

    phase = 'train' if train else 'test'
    info_fn = os.path.join(info_dir, 'Diving48_V2_{}.json'.format(phase))
    vocab_fn = os.path.join(info_dir, 'Diving48_vocab.json')
    sampling = 'random' if train else 'uniform'
    return Diving(data_dir, info_fn, vocab_fn, sampling=sampling, **kwargs)


def diving_fs(data_dir=find_path(DIVING_DATA),
              info_dir=find_path(DIVING_META),
              novel=False, base_cls=0, phase='all', **kwargs):

    n_cls = 48
    if base_cls > 0:
        subclass_fn = 'datasets/subclass/diving/{:d}.txt'.format(base_cls)
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

    info_fn = os.path.join(info_dir, 'Diving48_V2_{}.json'.format(phase))
    vocab_fn = os.path.join(info_dir, 'Diving48_vocab.json')
    sampling = 'random' if novel else 'uniform'
    return Diving(data_dir, info_fn, vocab_fn, classes=subclass_list, sampling=sampling, **kwargs)
