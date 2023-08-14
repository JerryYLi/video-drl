'''
Adversarial attackers
https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py
'''

import torch

def input_grad(inputs, targets, model, criterion):
    inputs.requires_grad = True
    output = model(inputs)
    loss = criterion(output, targets)
    grad = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
    inputs.requires_grad = False
    return grad


def var_grad(inputs, targets, model, criterion, var):
    output = model(inputs)
    loss = criterion(output, targets)
    grad = torch.autograd.grad(loss, var, retain_graph=True)[0]
    return grad


class FGSM:
    def __init__(self, eps, iters, momentum=0, scale=(1, 1, 1), dim=None):
        self.eps = eps  # maximum perturbation (l_inf)
        self.iters = iters
        self.momentum = momentum
        self.scale = scale
        self.dim = dim
    
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def perturb(self, inputs, targets, model, criterion):
        # scaling to account for normalization
        scale_shape = [3 if d == self.dim else 1 for d in range(len(inputs.shape))]
        scale = torch.tensor(self.scale, dtype=inputs.dtype, device=inputs.device).view(*scale_shape)

        # initialize attack example
        inputs_adv = inputs.clone()

        # attack iterations
        prev_grad = torch.zeros_like(inputs)
        for _ in range(self.iters):
            grad = input_grad(inputs_adv, targets, model, criterion)
            if self.momentum > 0:
                grad += self.momentum * prev_grad
                prev_grad = grad
            inputs_adv += self.eps / self.iters * grad.sign() / scale
        return inputs_adv


class SPFGSM:
    def __init__(self, eps, iters, momentum=0, scale=(1, 1, 1), dim=None, cp_dim=None):
        self.eps = eps  # maximum perturbation (l_inf)
        self.iters = iters
        self.momentum = momentum
        self.scale = scale
        self.dim = dim
        self.cp_dim = cp_dim
    
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def perturb(self, inputs, targets, model, criterion):
        # scaling to account for normalization
        scale_shape = [3 if d == self.dim else 1 for d in range(len(inputs.shape))]
        scale = torch.tensor(self.scale, dtype=inputs.dtype, device=inputs.device).view(*scale_shape)

        # initialize perturbation
        perturb_shape = list(inputs.shape)
        if self.cp_dim is not None:
            for k in self.cp_dim:
                perturb_shape[k] = 1
        z = torch.zeros(perturb_shape, dtype=inputs.dtype, device=inputs.device)

        # attack iterations
        prev_grad = torch.zeros_like(inputs)
        for _ in range(self.iters):
            z.requires_grad = True
            grad = var_grad(inputs + z, targets, model, criterion, var=z)
            z.requires_grad = False
            if self.momentum > 0:
                grad += self.momentum * prev_grad
                prev_grad = grad
            z += self.eps / self.iters * grad.sign() / scale
        return inputs + z


class PGD:
    def __init__(self, eps, alpha, iters, rand_start=True, scale=(1, 1, 1), dim=None):
        self.eps = eps  # maximum perturbation (l_inf)
        self.alpha = alpha  # step size
        self.iters = iters
        self.rand_start = rand_start
        self.scale = scale
        self.dim = dim
    
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def perturb(self, inputs, targets, model, criterion):
        # scaling to account for normalization
        scale_shape = [3 if d == self.dim else 1 for d in range(len(inputs.shape))]
        scale = torch.tensor(self.scale, dtype=inputs.dtype, device=inputs.device).view(*scale_shape)

        # initialize attack example
        inputs_adv = inputs.clone()
        if self.rand_start:
            rand_perturb = torch.rand_like(inputs) * 2 - 1
            inputs_adv += self.eps * rand_perturb / scale

        # attack iterations
        inputs_min = inputs - self.eps / scale
        inputs_max = inputs + self.eps / scale
        for _ in range(self.iters):
            grad = input_grad(inputs_adv, targets, model, criterion)
            inputs_adv += self.alpha * grad.sign() / scale
            inputs_adv = torch.min(torch.max(inputs_adv, inputs_min), inputs_max)

        return inputs_adv


class SPPGD:
    def __init__(self, eps, alpha, iters, rand_start=True, scale=(1, 1, 1), dim=None, cp_dim=None):
        self.eps = eps  # maximum perturbation (l_inf)
        self.alpha = alpha  # step size
        self.iters = iters
        self.rand_start = rand_start
        self.scale = scale
        self.dim = dim
        self.cp_dim = cp_dim
    
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def perturb(self, inputs, targets, model, criterion):
        # scaling to account for normalization
        scale_shape = [3 if d == self.dim else 1 for d in range(len(inputs.shape))]
        scale = torch.tensor(self.scale, dtype=inputs.dtype, device=inputs.device).view(*scale_shape)

        # initialize perturbation
        perturb_shape = list(inputs.shape)
        if self.cp_dim is not None:
            for k in self.cp_dim:
                perturb_shape[k] = 1
                
        if self.rand_start:
            z = torch.rand(perturb_shape, dtype=inputs.dtype, device=inputs.device) * 2 - 1
        else:
            z = torch.zeros(perturb_shape, dtype=inputs.dtype, device=inputs.device)

        # attack iterations
        max_perturb = self.eps / scale
        for _ in range(self.iters):
            z.requires_grad = True
            grad = var_grad(inputs + z, targets, model, criterion, var=z)
            z.requires_grad = False
            z += self.alpha * grad.sign() / scale
            z[z > max_perturb] = max_perturb
            z[z < -max_perturb] = -max_perturb

        return inputs + z
