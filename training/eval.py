import time
import torch
import torch.nn as nn
from .utils.metric import accuracy
from .utils.meter import AverageMeter, ClsAccMeter, ProgressMeter
from .utils.fsl import split_shot_query, make_nk_label
from models.fsl import CloserBaseline
    

def update_dict(results, batch):
    for k in batch:
        if k not in results:
            results[k] = batch[k]
        elif isinstance(batch[k], list):
            results[k].extend(batch[k])
        else:
            results[k] = torch.cat([results[k], batch[k]], 0)


def run_eval(model, model_sp, loader, criterion, criterion_sp, args, shuffle=False,
             metrics={'Acc@1': accuracy}, fprint=print, key='video'):
              
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    metric_meter = {**{k: AverageMeter(k, ':6.2f') for k in metrics}, **{k + '_sp': AverageMeter(k + '_sp', ':6.2f') for k in metrics}}
    cls_meter = ClsAccMeter()

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, loss_meter, *metric_meter.values()],
        fprint=fprint
    )

    # switch to train mode
    model.eval()
    if model_sp is not None:
        model_sp.eval()

    end = time.time()
    results = {}
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
        with torch.no_grad():
            outputs = model(inputs)
            if model_sp is not None:
                inputs_sp = inputs[0] if isinstance(inputs, list) else inputs
                outputs_sp = model_sp(inputs_sp)
            else:
                if shuffle:
                    if isinstance(inputs, list):
                        raise NotImplementedError
                    else:
                        idx = torch.randperm(inputs.shape[3])
                        inputs_sp = inputs[:, :, :, idx]  # shuffle frames
                else:
                    if isinstance(inputs, list):
                        inputs_sp = [x[:, :, :, :1].expand_as(x) for x in inputs]
                    else:
                        inputs_sp = inputs[:, :, :, :1].expand_as(inputs)  # freeze input video
                outputs_sp = model(inputs_sp)
            criterion.reduction = 'none'
            criterion_sp.reduction = 'none'
            loss = criterion(outputs, targets)
            loss_sp = criterion(outputs_sp, targets)
            loss_kd = criterion_sp(outputs_sp, outputs)
            pred = outputs.argmax(1)  # pseudo-labels

            # measure accuracy and record loss
            loss_meter.update(loss.mean().item(), bs)
            for k, met in metrics.items():
                metric_meter[k].update(met(outputs, targets)[0].item(), bs)
                metric_meter[k + '_sp'].update(met(outputs_sp, pred)[0].item(), bs)
            cls_meter.update(outputs, targets)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update results dict
        update_dict(results, {
            'vid': sample['vid'],
            'target': sample['label'],
            'output': outputs.cpu(),
            'output_sp': outputs_sp.cpu(),
            'loss': loss.cpu(),
            'loss_sp': loss_sp.cpu(), 
            'loss_kd': loss_kd.cpu(),
        })

        if args.debug and i >= 10:
            break

        if i % args.print_freq == 0 or i == len(loader) - 1:
            progress.display(i)
        
    return {k: metric_meter[k].avg for k in metric_meter}, cls_meter.accuracy(), results


def run_eval_fs(model, loader, criterion, args, 
                n_way, n_shot, n_query, ep_per_batch, 
                metrics={'Acc@1': accuracy}, fprint=print):
              
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    metric_meter = {k: AverageMeter(k, ':6.2f') for k in metrics}

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, loss_meter, *metric_meter.values()],
        fprint=fprint
    )

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        inputs = sample['video']
        inputs = inputs.cuda(args.gpu, non_blocking=True)

        # organize few-shot batch
        x_shot, x_query = split_shot_query(
            inputs, n_way, n_shot, n_query,
            ep_per_batch=ep_per_batch)
        targets = make_nk_label(n_way, n_query,
            ep_per_batch=ep_per_batch).cuda(args.gpu, non_blocking=True)
        
        # evaluate few-shot accuracy
        with torch.set_grad_enabled(isinstance(model, CloserBaseline)):
            # compute output
            outputs = model(x_shot, x_query).view(-1, n_way)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            loss_meter.update(loss.item(), inputs.shape[0])
            for k, met in metrics.items():
                metric_meter[k].update(met(outputs, targets)[0].item(), inputs.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(loader) - 1:
            progress.display(i)
        
    return {k: metric_meter[k].avg for k in metric_meter}