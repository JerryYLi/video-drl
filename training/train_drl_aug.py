import time
import torch
import torch.nn as nn
from .utils.metric import accuracy
from .utils.meter import AverageMeter, ProgressMeter

def get_feat_loss(pred, target, keys, crit):
    loss_dict = {}
    for k in keys:
        if pred[k].shape != target[k].shape:
            print(target[k].shape, pred[k].shape)
            target[k] = target[k].mean(3)
        assert pred[k].shape == target[k].shape
        loss_dict[k] = crit(pred[k], target[k].detach())
    return loss_dict


def run_epoch_drl_aug(epoch, train, model, model_sp, loader, adversary, 
                      criterion, criterion_sp, criterion_feat, optimizer, args,
                      feature=('pool',), evaluate=True, metrics={'Acc@1': accuracy},
                      fprint=print, writer=None, key='video'):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    metric_meter = {**{k: AverageMeter(k, ':6.2f') for k in metrics}, **{k + '_sp': AverageMeter(k + '_sp', ':6.2f') for k in metrics}}

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, loss_meter, *metric_meter.values()],
        prefix="Epoch: [{}]".format(epoch),
        fprint=fprint
    )

    # switch to train mode
    if evaluate:
        model.eval()
    else:
        model.train(train)
    model_sp.train(train)

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
            inputs_sp = inputs[0] if isinstance(inputs, list) else inputs
            outputs_sp = model_sp(inputs_sp)
            if criterion_feat is not None:
                loss_feat = get_feat_loss(outputs_sp, outputs, feature, criterion_feat)
                outputs, outputs_sp = outputs['pred'], outputs_sp['pred']
            loss_sp = criterion_sp(outputs_sp, outputs.detach())
            loss_cls = criterion(outputs, targets)
            pred = outputs.argmax(1)  # pseudo-labels

        # adversarial training (mix batch)
        if train and adversary is not None:
            model_sp.eval()
            inputs_adv = adversary.perturb(inputs_sp, pred, model_sp, criterion)
            model_sp.train()
            outputs_adv = model(inputs_adv)
            loss_adv = criterion(outputs_adv, targets)
            loss_cls = (loss_cls + loss_adv) / 2

        # measure accuracy and record loss
        loss = loss_cls + loss_sp
        if criterion_feat is not None:
            loss += sum(loss_feat.values())
        loss_meter.update(loss.item(), bs)
        for k, met in metrics.items():
            metric_meter[k].update(met(outputs, targets)[0].item(), bs)
            metric_meter[k + '_sp'].update(met(outputs_sp, pred)[0].item(), bs)
        
        if train and writer is not None:
            iters = epoch * len(loader) + i
            writer.add_scalar('train_iters/Loss', loss.item(), global_step=iters)
            writer.add_scalar('train_iters/Loss_sp', loss_sp.item(), global_step=iters)
            writer.add_scalar('train_iters/Loss_cls', loss_cls.item(), global_step=iters)
            if criterion_feat is not None:
                for k, v in loss_feat.items():
                    writer.add_scalar(f'train_iters/Loss_feat-{k}', v.item(), global_step=iters)
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