import os
import torch
import models
import datasets
import criteria
from datasets import video_transforms, volume_transforms

def build_model(cfg, gpu, args, ckp_key='state_dict', fprint=print):
    model = models.__dict__[cfg['name']](**cfg['args'])
    
    if 'sync_bn' in cfg and cfg['sync_bn']:
        assert args.ddp, 'sync batchnorm requires DistributedDataParallel'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        if args.ddp:
            args.workers = int((args.workers + torch.cuda.device_count() - 1) / torch.cuda.device_count())
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    elif args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    if 'pretrain' in cfg:
        path = cfg['pretrain']
        if os.path.isfile(path):
            if gpu is None:
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location='cuda:{}'.format(gpu))
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            pretrained_dict = checkpoint[ckp_key]
            model_dict = model.state_dict()
            # exclude head
            if 'copy_head' in cfg and cfg['copy_head']:
                fprint('copying model head')
            else:
                for k, v in model_dict.items():
                    if 'head' in k:
                        pretrained_dict[k] = v
            # TSM fix
            pretrained_dict = {
                k.replace('.base_model', '.net').replace('.new_fc', '.head'): v 
                for k, v in pretrained_dict.items()
            }
            model.load_state_dict(pretrained_dict)
            fprint("=> loaded checkpoint '{}' (epoch {})".format(path, epoch))
        else:
            fprint("=> no checkpoint found at '{}'".format(path))

    fprint(model)
    return model


def build_dataloader(cfg, args, subsample=None, fprint=print, debug=False):
    dataloaders = {}
    for k, db_cfg in cfg['args'].items():
        train = 'train' in k
        transform = build_transform(cfg['transform'], train=train)
        dataset = datasets.__dict__[cfg['name']](train=train, transform=transform, 
                                                 subsample=subsample if train else None, 
                                                 **db_cfg)
        fprint('{} split: {} videos'.format(k, len(dataset)))
        if debug:
            cfg['batch_size'] = 4
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cfg['batch_size'] = args.batch_size
        bs = 1 if 'final' in k else cfg['batch_size']
        if args.ddp and train:
            bs = int(bs / torch.cuda.device_count())
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloaders[k] = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=(train_sampler is None), 
                num_workers=args.workers, pin_memory=True, sampler=train_sampler
            )
        else:
            dataloaders[k] = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=train, 
                num_workers=args.workers, pin_memory=True
            )
    return dataloaders


def build_transform(cfg, train):
    transform_list = []
    
    if train:
        if 'rnd_scale' in cfg:
            transform_list.append(video_transforms.RandomResizedCrop(cfg['crop'], 
                scale=cfg['rnd_scale'] if 'rnd_scale' in cfg else (0.08, 1.0), 
                ratio=cfg['rnd_ratio'] if 'rnd_ratio' in cfg else (3. / 4., 4. / 3.)))
        else:
            transform_list.append(video_transforms.Resize(cfg['resize'], interpolation='bilinear'))
            if 'crop' in cfg:
                transform_list.append(video_transforms.RandomCrop(cfg['crop']))
        if 'hflip' in cfg and cfg['hflip']:
            transform_list.append(video_transforms.RandomHorizontalFlip())
        if 'color_jitter' in cfg:
            transform_list.append(video_transforms.ColorJitter(*cfg['color_jitter']))
    else:
        transform_list.append(video_transforms.Resize(cfg['resize'], interpolation='bilinear'))
        if 'crop' in cfg:
            transform_list.append(video_transforms.CenterCrop(cfg['crop']))
    
    transform_list.append(volume_transforms.ClipToTensor())

    if 'normalize' in cfg:
        transform_list.append(
            video_transforms.Normalize(**cfg['normalize'])
        )

    return video_transforms.Compose(transform_list)


def build_optimizer(cfg, model, fprint=print):
    def get_module(model, module_name):
        if isinstance(model, torch.nn.DataParallel):
            return get_module(model.module, module_name)
        if '.' in module_name:
            sub, module_name = module_name.split('.', 1)
            return get_module(getattr(model, sub), module_name)
        return getattr(model, module_name)

    def build_param_group(model, cfg):
        if 'params' in cfg:
            params = sum([list(get_module(model, m).parameters()) for m in cfg['params']], [])
        else:
            params = list(model.parameters())
        param_group = {'params': params, **cfg['args']}
        return param_group

    optimizer = torch.optim.__dict__[cfg['name']]([
        build_param_group(model, cfg)
    ], **cfg['args'])
    
    if 'schedule' in cfg:
        scheduler = torch.optim.lr_scheduler.__dict__[cfg['schedule']['name']](
            optimizer, **cfg['schedule']['args']
        )
    else:
        scheduler = None

    fprint(optimizer)
    return optimizer, scheduler


def build_criterion(cfg, fprint=print):
    if cfg['name'] in torch.nn.__dict__:
        criterion = torch.nn.__dict__[cfg['name']](**cfg['args'])
    else:
        criterion = criteria.__dict__[cfg['name']](**cfg['args'])
    fprint(criterion)
    return criterion


def print_line(msg, path):
    if path is not None:
        with open(path, 'a') as f:
            f.write(str(msg) + '\n')
    else:
        print(msg)