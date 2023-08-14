'''
Dynamic representation learning by adversarial augmentation
'''

import torch
import random
import os
import sys
import yaml
import argparse
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp

from training import run_epoch_drl_aug, resume_checkpoint_dual, save_checkpoint_dual, adversarial
import training.utils.base as base_utils
import training.utils.adv as adv_utils

parser = argparse.ArgumentParser(description='Video recognition training')
parser.add_argument('cfg', metavar='CFG', 
                    help='path to config file of main model')
parser.add_argument('sp_cfg', metavar='SP_CFG', 
                    help='path to config file of spatial model')
parser.add_argument('--no-resume', dest='resume', action='store_false',
                    help='resume from checkpoint')
parser.add_argument('--log', action='store_true',
                    help='save training log')
parser.add_argument('--pretrain', default=None, type=str,
                    help='Path to pretrained model')
parser.add_argument('--scratch', action='store_true',
                    help='do not load pretrained model')
parser.add_argument('--ckp', default='checkpoint.pth.tar', type=str,
                    help='Name of checkpoint file')
parser.add_argument('-b', '--batch-size', default=None, type=int,
                    help='override batch size')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--debug', action='store_true',
                    help='use smaller batch size for debugging')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')


def main():
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(args.sp_cfg, 'r') as f:
        sp_cfg = yaml.safe_load(f)

    if 'seed' in cfg:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # checkpoint, log dir
    ckp_dir = get_ckp(cfg, args)
    os.makedirs(ckp_dir, exist_ok=True)

    args.ddp = 'ddp' in cfg['model'] and cfg['model']['ddp']
    if args.ddp:
        mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(ckp_dir, cfg, sp_cfg, args))
    else:
        main_worker(args.gpu, ckp_dir, cfg, sp_cfg, args)


def main_worker(gpu, ckp_dir, cfg, sp_cfg, args):
    log_path = os.path.join(ckp_dir, 'train.log') if args.log else None
    fprint = partial(base_utils.print_line, path=log_path)

    log_dir = os.path.join(ckp_dir, 'runs')
    writer = SummaryWriter(log_dir)

    if args.ddp:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=torch.cuda.device_count(), rank=gpu)

    # prepare model, optimizer, dataset
    if sp_cfg['model'] is not None:
        sp_cfg['model']['args']['n_class'] = cfg['model']['args']['n_class']
        if 'ddp' in cfg['model']:
            sp_cfg['model']['ddp'] = cfg['model']['ddp']
        if 'sync_bn' in cfg['model']:
            sp_cfg['model']['sync_bn'] = cfg['model']['sync_bn']
        if 'criterion_feat' in sp_cfg:
            sp_cfg['model']['args']['ret_feat'] = sp_cfg['criterion_feat']['keys']
            cfg['model']['args']['ret_feat'] = sp_cfg['criterion_feat']['keys']
    
    model = base_utils.build_model(cfg['model'], gpu, args, fprint=fprint)
    model_sp = base_utils.build_model(sp_cfg['model'], gpu, args, fprint=fprint) if sp_cfg['model'] is not None else None
    adversary = adv_utils.build_adversary(sp_cfg['adversary'], fprint=fprint) if 'adversary' in sp_cfg else None
    optimizer, scheduler = adv_utils.build_optimizer(sp_cfg['optimizer'], model, model_sp, fprint=fprint)
    criterion = base_utils.build_criterion(cfg['criterion'], fprint=fprint)
    criterion_sp = base_utils.build_criterion(sp_cfg['criterion'], fprint=fprint)
    criterion_feat = base_utils.build_criterion(sp_cfg['criterion_feat'], fprint=fprint) if 'criterion_feat' in sp_cfg else None
    feature = sp_cfg['criterion_feat']['keys'] if 'criterion_feat' in sp_cfg else None
    dataloaders = base_utils.build_dataloader(cfg['dataset'], args, fprint=fprint, debug=args.debug)
    
    start_epoch, best_acc1 = 0, 0
    if args.resume:
        start_epoch, best_acc1 = resume_checkpoint_dual(ckp_dir, model, model_sp, optimizer, scheduler, args.gpu, fprint=fprint)

    torch.backends.cudnn.benchmark = True

    # train model
    epoch = start_epoch
    if not args.evaluate:
        for epoch in range(start_epoch, sp_cfg['optimizer']['epochs']):
            if args.ddp:
                dataloaders['train'].sampler.set_epoch(epoch)
            use_adv = 'adversary' in cfg and not ('epoch_start' in cfg['adversary'] and epoch < cfg['adversary']['epoch_start'])
            run_epoch_drl_aug(epoch, True, model, model_sp, dataloaders['train'], adversary if use_adv else None, 
                              criterion, criterion_sp, criterion_feat, optimizer, args, 
                              feature=feature, evaluate=not use_adv, fprint=fprint, writer=writer)
            
            # save checkpoint
            if not args.ddp or gpu == 0:
                acc1 = run_epoch_drl_aug(epoch, False, model, model_sp, dataloaders['test'], adversary if use_adv else None, 
                                         criterion, criterion_sp, criterion_feat, None, args, 
                                         feature=feature, fprint=fprint, writer=writer)['Acc@1']
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                save_checkpoint_dual(epoch, model, model_sp, optimizer, scheduler, best_acc1, is_best, ckp_dir)

            if scheduler is not None:
                scheduler.step()

    # final test
    if not args.ddp or gpu == 0:
        met = run_epoch_drl_aug(epoch, False, model, model_sp, dataloaders['test_final'], None, 
                                criterion, criterion_sp, criterion_feat, None, args, 
                                feature=feature, fprint=fprint)
        fprint('Acc@1: {:6.2f}%'.format(met['Acc@1']))
        fprint('Acc@1_sp: {:6.2f}%'.format(met['Acc@1_sp']))

    if args.ddp:
        dist.destroy_process_group()


def get_ckp(cfg, args):
    if args.pretrain is not None:
        pre_dir = os.path.dirname(args.pretrain)
        cfg['model']['pretrain'] = args.pretrain
    elif not args.scratch:
        pre_dir = args.cfg.replace('.yaml', '').replace('configs', 'checkpoints')
        cfg['model']['pretrain'] = os.path.join(pre_dir, args.ckp)
    cfg['model']['copy_head'] = True
    ckp_dir = os.path.join(pre_dir, args.sp_cfg.replace('.yaml', '').split('/', 1)[1])
    if args.debug:
        ckp_dir += '_debug'
    return ckp_dir


if __name__ == '__main__':
    main()