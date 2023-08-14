import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_bin(output, target):
    """Computes the binary classification accuracy """
    with torch.no_grad():
        assert output.shape == target.shape
        batch_size = target.size(0)
        correct = (output > 0) == target
        acc = torch.all(correct, 1).float().mean()
        return [acc]
