import torch
import models
import datasets
import criteria
from datasets import video_transforms, volume_transforms
from training import adversarial

def build_adversary(cfg, fprint=print):
    adversary = adversarial.__dict__[cfg['name']](**cfg['args'])
    fprint(adversary)
    return adversary


def build_optimizer(cfg, model, model_sp, fprint=print):

    def build_param_group(model, cfg):
        if 'params' in cfg:
            if isinstance(model, torch.nn.DataParallel):
                params = sum([list(getattr(model.module, m).parameters()) for m in cfg['params']], [])
            else:
                params = sum([list(getattr(model, m).parameters()) for m in cfg['params']])
        else:
            params = list(model.parameters())
        param_group = {'params': params, **cfg['args']}
        return param_group

    param_groups = []
    if model_sp is not None:
        param_groups.append(build_param_group(model_sp, cfg['model_sp']))
    if 'model' in cfg:
        param_groups.append(build_param_group(model, cfg['model']))
    optimizer = torch.optim.__dict__[cfg['name']](param_groups, **cfg['model_sp']['args'])
    
    if 'schedule' in cfg:
        scheduler = torch.optim.lr_scheduler.__dict__[cfg['schedule']['name']](
            optimizer, **cfg['schedule']['args']
        )
    else:
        scheduler = None

    fprint(optimizer)
    return optimizer, scheduler