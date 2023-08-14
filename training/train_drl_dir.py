import time
import torch
import torch.nn as nn
from .utils.metric import accuracy
from .utils.meter import AverageMeter, ProgressMeter

class RevGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


class RevGrad(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return RevGradFn.apply(input_)


def run_epoch_drl_dir(epoch, train, model, model_sp, loader, lambda_sp,
                      criterion, criterion_sp, optimizer, args, scale=False, shuffle=False,
                      metrics={'Acc@1': accuracy}, fprint=print, writer=None,
                      freeze_bn=False, key='video'):
              
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    metric_meter = {**{k: AverageMeter(k, ':6.2f') for k in metrics}, **{k + '_sp': AverageMeter(k, ':6.2f') for k in metrics}}

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
                
    if model_sp is not None:
        model_sp.train(train)

    if train:
        rev_grad = RevGrad()
    
    if scale:
        criterion.reduction = 'none'

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
            if model_sp is not None:
                inputs_sp = inputs[0] if isinstance(inputs, list) else inputs
                outputs_sp = model_sp(inputs_sp)
                if train:
                    outputs_sp = rev_grad(outputs_sp)
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
                outputs_sp = model(inputs_sp).detach()
            loss_sp = criterion_sp(outputs_sp, outputs)
            loss_cls = criterion(outputs, targets)
            loss_cls_sp = criterion(outputs_sp, targets)
            pred = outputs.argmax(1)  # pseudo-labels

        # measure accuracy and record loss
        if scale:
            ts = 1 - torch.exp(-torch.relu(loss_cls_sp - loss_cls))
            loss_cls = loss_cls.mean()
            loss_cls_sp = loss_cls_sp.mean()
            loss = loss_cls - lambda_sp * (ts.detach() * loss_sp).mean()
        else:
            loss = loss_cls - lambda_sp * loss_sp
        if model_sp is not None:
            loss -= loss_cls_sp
        loss_meter.update(loss.item(), bs)
        for k, met in metrics.items():
            metric_meter[k].update(met(outputs, targets)[0].item(), bs)
            metric_meter[k + '_sp'].update(met(outputs_sp, pred)[0].item(), bs)
        
        if train and writer is not None:
            iters = epoch * len(loader) + i
            writer.add_scalar('train_iters/Loss', loss.item(), global_step=iters)
            writer.add_scalar('train_iters/Loss_sp', loss_sp.item(), global_step=iters)
            writer.add_scalar('train_iters/Loss_cls', loss_cls.item(), global_step=iters)
            writer.add_scalar('train_iters/Loss_cls-sp', loss_cls_sp.item(), global_step=iters)
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