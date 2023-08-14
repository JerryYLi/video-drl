'''
Video network training
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

from training import run_epoch, resume_checkpoint, save_checkpoint
import training.utils.base as base_utils
import training.utils.adv as adv_utils

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
parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')


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

    args.ddp = 'ddp' in cfg['model'] and cfg['model']['ddp']
    if args.ddp:
        mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(ckp_dir, cfg, args))
    else:
        main_worker(args.gpu, ckp_dir, cfg, args)


def main_worker(gpu, ckp_dir, cfg, args):
    log_path = os.path.join(ckp_dir, 'train.log') if args.log else None
    fprint = partial(base_utils.print_line, path=log_path)

    log_dir = os.path.join(ckp_dir, 'runs')
    writer = SummaryWriter(log_dir)

    if args.ddp:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=torch.cuda.device_count(), rank=gpu)

    # prepare model, optimizer, dataset
    model = base_utils.build_model(cfg['model'], gpu, args, fprint=fprint)
    optimizer, scheduler = base_utils.build_optimizer(cfg['optimizer'], model, fprint=fprint)
    criterion = base_utils.build_criterion(cfg['criterion'], fprint=fprint)
    dataloaders = base_utils.build_dataloader(cfg['dataset'], args, fprint=fprint, debug=args.debug)
    
    start_epoch, best_acc1 = 0, 0
    if args.resume:
        start_epoch, best_acc1 = resume_checkpoint(ckp_dir, model, optimizer, scheduler, args.gpu, fprint=fprint)

    torch.backends.cudnn.benchmark = True

    # train model
    epoch = start_epoch
    if not args.evaluate:
        for epoch in range(start_epoch, cfg['optimizer']['epochs']):
            freeze_bn = cfg['model']['freeze_bn'] if 'freeze_bn' in cfg['model'] else False
            validate = not args.ddp or gpu == 0  # only validate on 1st gpu with ddp
            if args.ddp:
                dataloaders['train'].sampler.set_epoch(epoch)
            run_epoch(epoch, True, model, dataloaders['train'], criterion, optimizer, args,
                      fprint=fprint, writer=writer if validate else None, freeze_bn=freeze_bn)

            # save checkpoint
            if validate:
                acc1 = run_epoch(epoch, False, model, dataloaders['test'], criterion, None, args, fprint=fprint, writer=writer)['Acc@1']
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                save_checkpoint(epoch, model, optimizer, scheduler, best_acc1, is_best, ckp_dir)

            if scheduler is not None:
                scheduler.step()

    # final test
    if validate:
        acc1 = run_epoch(epoch, False, model, dataloaders['test_final'], criterion, None, args, fprint=fprint)['Acc@1']
        fprint('Acc@1: {:6.2f}%'.format(acc1))

    if args.ddp:
        dist.destroy_process_group()


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