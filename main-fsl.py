'''
Video few-shot classification
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

from training import run_epoch, run_eval_fs
from training.utils.base import build_model, build_optimizer, build_criterion, print_line
from training.utils.fsl import build_dataloader, build_model_fs

parser = argparse.ArgumentParser(description='Video recognition training')
parser.add_argument('cfg', metavar='CFG', 
                    help='path to config file')
parser.add_argument('--no-resume', dest='resume', action='store_false',
                    help='resume from checkpoint')
parser.add_argument('--log', action='store_true',
                    help='save training log')
parser.add_argument('--pretrain', default=None, type=str,
                    help='Path to pretrained model')
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


def main():
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'seed' in cfg:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    ckp_dir = get_ckp(cfg, args)
    os.makedirs(ckp_dir, exist_ok=True)

    log_path = os.path.join(ckp_dir, 'train.log') if args.log else None
    fprint = partial(print_line, path=log_path)

    log_dir = os.path.join(ckp_dir, 'runs')
    writer = SummaryWriter(log_dir)

    # prepare model, dataset
    cfg['model']['ddp'] = False
    cfg['model']['sync_bn'] = False
    args.ddp = False
    model = build_model(cfg['model'], args.gpu, args, fprint=fprint)
    if 'optimizer' in cfg:
        optimizer, scheduler = build_optimizer(cfg['optimizer'], model, fprint=fprint)
    else:
        fprint('Optimizer config not found. Setting evaluate=True')
        args.evaluate = True
    
    criterion = build_criterion(cfg['criterion'], fprint=fprint)
    dataloaders = build_dataloader(cfg['dataset'], args, fprint=fprint, debug=args.debug)

    torch.backends.cudnn.benchmark = True

    # train model
    if not args.evaluate:
        for epoch in range(cfg['optimizer']['epochs']):
            freeze_bn = cfg['model']['freeze_bn'] if 'freeze_bn' in cfg['model'] else False
            run_epoch(epoch, True, model, dataloaders['base'], criterion, optimizer, args,
                      fprint=fprint, writer=writer, freeze_bn=freeze_bn)

            if scheduler is not None:
                scheduler.step()

    # final test
    n_way = cfg['dataset']['n_way']
    n_shot = cfg['dataset']['n_shot']
    n_query = cfg['dataset']['n_query']
    ep_per_batch = cfg['dataset']['ep_per_batch']
    model = build_model_fs(cfg['model_fs'], model)
    acc1 = run_eval_fs(model, dataloaders['novel'], criterion, args, 
                       n_way, n_shot, n_query, ep_per_batch, fprint=fprint)['Acc@1']
    fprint('Acc@1: {:6.2f}%'.format(acc1))


def get_ckp(cfg, args):
    if args.pretrain is not None:
        cfg['model']['pretrain'] = args.pretrain
    if 'pretrain' in cfg['model']:
        pre_dir = os.path.dirname(cfg['model']['pretrain'])
        ckp_dir = os.path.join(pre_dir, args.cfg.replace('.yaml', '').split('/', 1)[1])
    else:
        ckp_dir = args.cfg.replace('.yaml', '').replace('configs', 'checkpoints')
    if args.debug:
        ckp_dir += '_debug'
    return ckp_dir


if __name__ == '__main__':
    main()