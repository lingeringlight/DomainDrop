import torch
from torchvision.utils import save_image
import numpy as np

def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def mean_teacher(model, teacher, momentum=0.9995):
    model_dict = model.state_dict()
    teacher_dict = teacher.state_dict()
    for k, v in teacher_dict.items():
        teacher_dict[k] = v * momentum + (1 - momentum) * model_dict[k]

    teacher.load_state_dict(teacher_dict)


def update_teacher(model, teacher, momentum=0.9995):
    for ema_param, param in zip(teacher.parameters(), model.parameters()):
        ema_param.data.mul_(momentum).add_(1 - momentum, param.data)


def warm_update_teacher(model, teacher, momentum=0.9995, global_step=2000):
    momentum = min(1 - 1 / (global_step + 1), momentum)
    for ema_param, param in zip(teacher.parameters(), model.parameters()):
        ema_param.data.mul_(momentum).add_(1 - momentum, param.data)


def preprocess_teacher(model, teacher):
    for param_m, param_t in zip(model.parameters(), teacher.parameters()):
        param_t.data.copy_(param_m.data)  # initialize
        param_t.requires_grad = False  # not update by gradient


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)


def update_bn(loader, model, device=None, swa_bn_domaindrop=0):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            data = input[0][0]
            class_l = input[0][2]
            domain_labels = input[0][3]
        if device is not None:
            data = data.to(device)
            class_l = class_l.to(device)
            domain_labels = domain_labels.to(device)

        model(data, gt=class_l, domain_labels=domain_labels, swa_bn_domaindrop=swa_bn_domaindrop)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)