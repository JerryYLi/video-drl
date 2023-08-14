import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    r"""Knowledge Distillation Loss based on KL Divergence

    # Input is expected to contain log-probabilities, targets are given as probabilities.
    Input and target both expected to be logits

    Args:
        temperature (float, optional): Softmax temperature. Default: 1
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied.
            ``'batchmean'``: the sum of the output will be divided by batchsize.
            ``'sum'``: the output will be summed.
            ``'mean'``: the output will be divided by the number of elements in the output.
            Default: ``'batchmean'``

    Shape:
        - Input: :math:`(N, C)`
        - Target: :math:`(N, C)`, same shape as the input
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, C)`,
          the same shape as the input

    """
    def __init__(self, temperature=1, reduction='batchmean'):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, input, target):
        assert len(input.shape) == 2, 'wrong input shape {}'.format(input.shape)
        assert input.shape == target.shape, 'input {} and target {} mismatch'.format(input.shape, target.shape)
        input_logp = F.log_softmax(input / self.temperature, dim=1)
        target_p = F.softmax(target / self.temperature, dim=1)
        return F.kl_div(input_logp, target_p, reduction=self.reduction) * (self.temperature ** 2)


class JSDLoss(nn.Module):
    '''
    JS divergence between probability distributions
    '''
    def __init__(self, temperature=1, reduction='batchmean'):
        super(JSDLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, input, target):
        assert len(input.shape) == 2, 'wrong input shape {}'.format(input.shape)
        assert input.shape == target.shape, 'input {} and target {} mismatch'.format(input.shape, target.shape)
        input_p = F.softmax(input / self.temperature, dim=1)
        target_p = F.softmax(target / self.temperature, dim=1)
        mean_p = (input_p + target_p) / 2
        kl1 = F.kl_div(input_p.log(), mean_p, reduction=self.reduction) * (self.temperature ** 2)
        kl2 = F.kl_div(target_p.log(), mean_p, reduction=self.reduction) * (self.temperature ** 2)
        return (kl1 + kl2) / 2