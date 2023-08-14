import time
import torch
import torch.nn as nn
from .utils.metric import accuracy, accuracy_bin
from .utils.meter import AverageMeter, ProgressMeter

def run_epoch(epoch, train, model, loader, criterion, optimizer, args, 
              metrics={'Acc@1': accuracy}, fprint=print, writer=None, 
              freeze_bn=False, key='video'):

    # use binary accuracy for multi-label loss
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        metrics = {'Acc@1': accuracy_bin}
              
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    metric_meter = {k: AverageMeter(k, ':6.2f') for k in metrics}

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, loss_meter, *metric_meter.values()],
        prefix="Epoch: [{}]".format(epoch),
        fprint=fprint
    )

    # switch to train mode
    model.train(train)
    if freeze_bn:
        for m in model.modules():
            if 'BatchNorm' in m.__class__.__name__:
                m.eval()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if isinstance(key, list):
            inputs = [sample[k].cuda(args.gpu, non_blocking=True) for k in key]
            bs = inputs[0].shape[0]
        else:
            inputs = sample[key].cuda(args.gpu, non_blocking=True)
            bs = inputs.shape[0]
        targets = sample['label'].cuda(args.gpu, non_blocking=True)
        
        # compute output
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs['pred']
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        loss_meter.update(loss.item(), bs)
        for k, met in metrics.items():
            metric_meter[k].update(met(outputs, targets)[0].item(), bs)
        
        if train and writer is not None:
            iters = epoch * len(loader) + i
            writer.add_scalar('train_iters/Loss', loss.item(), global_step=iters)
            for k, met in metric_meter.items():
                writer.add_scalar('train_iters/{}'.format(k), met.val, global_step=iters)

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(loader) - 1:
            progress.display(i)

    # log results
    if writer is not None:
        phase = 'train' if train else 'val'
        writer.add_scalar('{}_epochs/Loss'.format(phase), loss_meter.avg, global_step=epoch)
        for k, met in metric_meter.items():
            writer.add_scalar('{}_epochs/{}'.format(phase, k), met.avg, global_step=epoch)
        
    return {k: metric_meter[k].avg for k in metric_meter}