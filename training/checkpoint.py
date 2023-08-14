import os
import shutil
import torch


def resume_checkpoint(dir, model, optimizer, scheduler, gpu, filename='checkpoint.pth.tar', key='state_dict', fprint=print):
    path = os.path.join(dir, filename)
    if os.path.isfile(path):
        if gpu is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cuda:{}'.format(gpu))
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint[key])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        fprint("=> loaded checkpoint '{}' (epoch {})"
                .format(path, start_epoch))
    else:
        start_epoch, best_acc1 = 0, 0
        fprint("=> no checkpoint found at '{}'".format(path))
                
    return start_epoch, best_acc1


def save_checkpoint(epoch, model, optimizer, scheduler, best_acc1, is_best, dir, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_acc1': best_acc1,
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    save_path = os.path.join(dir, filename)
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(dir, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_path)


def resume_checkpoint_dual(dir, model, model_sp, optimizer, scheduler, gpu, filename='checkpoint.pth.tar', fprint=print):
    path = os.path.join(dir, filename)
    if os.path.isfile(path):
        if gpu is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cuda:{}'.format(gpu))
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        if model_sp is not None:
            model_sp.load_state_dict(checkpoint['state_dict_sp'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        fprint("=> loaded checkpoint '{}' (epoch {})"
                .format(path, start_epoch))
    else:
        start_epoch, best_acc1 = 0, 0
        fprint("=> no checkpoint found at '{}'".format(path))
                
    return start_epoch, best_acc1


def save_checkpoint_dual(epoch, model, model_sp, optimizer, scheduler, best_acc1, is_best, dir, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_acc1': best_acc1,
    }
    if model_sp is not None:
        state['state_dict_sp'] = model_sp.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    save_path = os.path.join(dir, filename)
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(dir, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_path)