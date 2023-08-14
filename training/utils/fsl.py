'''
Few-shot learning utils
adapted from https://github.com/yinboc/few-shot-meta-baseline/
'''

import os
import torch
import numpy as np
import models
import datasets
import criteria
from datasets import video_transforms, volume_transforms
import models.fsl as model_fs
from .base import build_transform

class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1, min_exps=5):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            catloc = np.argwhere(label == c).reshape(-1)
            if len(catloc) >= min_exps:
                self.catlocs.append(catloc)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            for _ in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)


def build_dataloader(cfg, args, fprint=print, debug=False):
    dataloaders = {}
    for k, db_cfg in cfg['args'].items():
        novel = 'novel' in k
        transform = build_transform(cfg['transform'], train=not novel)
        dataset = datasets.__dict__[cfg['name']](novel=novel, transform=transform, **db_cfg)
        fprint('{} split: {} videos'.format(k, len(dataset)))
        if novel:
            batch_sampler = CategoriesSampler(dataset.data['label'], cfg['n_batch'], cfg['n_way'], 
                                              cfg['n_shot'] + cfg['n_query'], ep_per_batch=cfg['ep_per_batch'])
            num_workers = min(4, args.workers)
            dataloaders[k] = torch.utils.data.DataLoader(dataset, 
                                                         batch_sampler=batch_sampler, 
                                                         num_workers=num_workers, 
                                                         pin_memory=True)
        else:
            bs = cfg['batch_size'] if not debug else 4
            dataloaders[k] = torch.utils.data.DataLoader(dataset, 
                                                         batch_size=bs, 
                                                         shuffle=True, 
                                                         num_workers=args.workers, 
                                                         pin_memory=True)

    return dataloaders


def build_model_fs(cfg, model):
    model = model_fs.__dict__[cfg['name']](model, **cfg['args'])
    return model


def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label
