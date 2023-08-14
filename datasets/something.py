import os
import re
import csv
import json
import torch
import random
from torchvision import transforms

from .video_dataset import VideoDataset
from ._paths import *
from .utils.torch_videovision.torchvideotransforms import video_transforms, volume_transforms


class SomethingV1(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, **kwargs):
        data = {}

        with open(split_file, 'r') as f:
            data_info = list(csv.reader(f, delimiter=';'))
            data['video'] = [os.path.join(data_dir, vid) for vid, _ in data_info]
            data['vid'] = [vid for vid, _ in data_info]
        with open(vocab_file, 'r') as f:
            vocab = list(map(lambda x: x.rstrip(), f))
            data['label'] = [vocab.index(label) for _, label in data_info]
        
        super().__init__(data, loader='VideoFramesLoader', **kwargs)
        self.vocab = vocab


class SomethingV1FS(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, classes=None, **kwargs):
        data = {}

        with open(split_file, 'r') as f:
            data_info = list(csv.reader(f, delimiter=';'))
        with open(vocab_file, 'r') as f:
            vocab = list(map(lambda x: x.rstrip(), f))
        data['video'] = []
        data['vid'] = []
        data['label'] = []
        for vid, label in data_info:
            label_id = vocab.index(label)
            if classes is not None:
                if label_id in classes:
                    label_id = classes.index(label_id)
                else:
                    continue
            data['video'].append(os.path.join(data_dir, vid))
            data['vid'].append(vid)
            data['label'].append(label_id)
        
        super().__init__(data, loader='VideoFramesLoader', **kwargs)
        self.vocab = vocab


class SomethingV1_Test(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, **kwargs):
        data = {}

        with open(split_file, 'r') as f:
            data['vid'] = [vid for vid, in csv.reader(f)]
            data['video'] = [os.path.join(data_dir, vid) for vid in data['vid']]
        with open(vocab_file, 'r') as f:
            vocab = list(map(lambda x: x.rstrip(), f))
        
        super().__init__(data, loader='VideoFramesLoader', **kwargs)
        self.vocab = vocab


class SomethingV2(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, **kwargs):
        data = {}

        with open(split_file, 'r') as f:
            data_info = json.load(f)
            data['video'] = [os.path.join(data_dir, vid['id'] + '.webm') for vid in data_info]
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
            data['label'] = [int(vocab[re.sub('[\[\]]', '', vid['template'])]) for vid in data_info]
        
        super().__init__(data, **kwargs)
        self.vocab = list(vocab.keys())


class SomethingV2FS(VideoDataset):
    def __init__(self, data_dir, split_file, vocab_file, classes=None, **kwargs):
        data = {}

        with open(split_file, 'r') as f:
            data_info = json.load(f)
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        data['video'] = []
        data['vid'] = []
        data['label'] = []
        for vid in data_info:
            label_id = int(vocab[re.sub('[\[\]]', '', vid['template'])])
            if classes is not None:
                if label_id in classes:
                    label_id = classes.index(label_id)
                else:
                    continue
            data['video'].append(os.path.join(data_dir, vid['id'] + '.webm'))
            data['vid'].append(vid['id'])
            data['label'].append(label_id)

        super().__init__(data, **kwargs)
        self.vocab = list(vocab.keys())


def something_v1(data_dir=find_path(STHV1_DATA),
                 info_dir=find_path(STHV1_META),
                 train=True, **kwargs):

    phase = 'train' if train else 'validation'
    split_path = os.path.join(info_dir, 'something-something-v1-%s.csv' % phase)
    vocab_path = os.path.join(info_dir, 'something-something-v1-labels.csv')
    sampling = 'random' if train else 'uniform'
    return SomethingV1(data_dir, split_path, vocab_path, sampling=sampling, **kwargs)


def something_v1_test(data_dir=find_path(STHV1_DATA),
                      info_dir=find_path(STHV1_META),
                      **kwargs):

    split_path = os.path.join(info_dir, 'something-something-v1-test.csv')
    vocab_path = os.path.join(info_dir, 'something-something-v1-labels.csv')
    return SomethingV1_Test(data_dir, split_path, vocab_path, sampling='uniform', **kwargs)


def something_v1_fs(data_dir=find_path(STHV1_DATA),
                    info_dir=find_path(STHV1_META),
                    novel=False, base_cls=0, phase='validation', **kwargs):

    n_cls = 174
    if base_cls > 0:
        subclass_fn = 'datasets/subclass/sthv1/{:d}.txt'.format(base_cls)
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
    split_path = os.path.join(info_dir, 'something-something-v1-%s.csv' % phase)
    vocab_path = os.path.join(info_dir, 'something-something-v1-labels.csv')
    return SomethingV1FS(data_dir, split_path, vocab_path, classes=subclass_list, sampling=sampling, **kwargs)


def something_v2(data_dir=find_path(STHV2_DATA),
                 info_dir=find_path(STHV2_META),
                 train=True, **kwargs):

    phase = 'train' if train else 'validation'
    split_path = os.path.join(info_dir, 'something-something-v2-%s.json' % phase)
    vocab_path = os.path.join(info_dir, 'something-something-v2-labels.json')
    sampling = 'random' if train else 'uniform'
    return SomethingV2(data_dir, split_path, vocab_path, sampling=sampling, **kwargs)


def something_v2_fs(data_dir=find_path(STHV2_DATA),
                    info_dir=find_path(STHV2_META),
                    novel=False, base_cls=0, phase='validation', **kwargs):

    n_cls = 174
    if base_cls > 0:
        subclass_fn = 'datasets/subclass/sthv2/{:d}.txt'.format(base_cls)
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
    split_path = os.path.join(info_dir, 'something-something-v2-%s.json' % phase)
    vocab_path = os.path.join(info_dir, 'something-something-v2-labels.json')
    return SomethingV2FS(data_dir, split_path, vocab_path, classes=subclass_list, sampling=sampling, **kwargs)


def jester(data_dir=find_path(JESTER_DATA),
           info_dir=find_path(JESTER_META),
           train=True, **kwargs):

    phase = 'train' if train else 'validation'
    split_path = os.path.join(info_dir, 'jester-v1-%s.csv' % phase)
    vocab_path = os.path.join(info_dir, 'jester-v1-labels.csv')
    sampling = 'random' if train else 'uniform'
    return SomethingV1(data_dir, split_path, vocab_path, sampling=sampling, **kwargs)


def jester_fs(data_dir=find_path(JESTER_DATA),
              info_dir=find_path(JESTER_META),
              novel=False, base_cls=0, phase='validation', **kwargs):

    n_cls = 27
    if base_cls > 0:
        subclass_fn = 'datasets/subclass/jester/{:d}.txt'.format(base_cls)
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
    split_path = os.path.join(info_dir, 'jester-v1-%s.csv' % phase)
    vocab_path = os.path.join(info_dir, 'jester-v1-labels.csv')
    return SomethingV1FS(data_dir, split_path, vocab_path, classes=subclass_list, sampling=sampling, **kwargs)