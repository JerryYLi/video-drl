import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ClsAccMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.reset()

    def reset(self):
        self.cls_count = torch.zeros(self.num_classes)
        self.cls_correct = torch.zeros(self.num_classes)

    def update(self, output, target):
        n = target.size(0)
        if self.num_classes is None:
            self.num_classes = output.size(1)
            self.reset()
        pred = output.argmax(1).unsqueeze(0)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for k in range(self.num_classes):
            self.cls_count[k] += (target == k).float().sum().cpu()
            self.cls_correct[k] += correct[target == k].float().sum().cpu()
        
    def accuracy(self):
        cls_accuracy = self.cls_correct / self.cls_count
        return cls_accuracy


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fprint=print):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fprint = fprint

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.fprint('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'