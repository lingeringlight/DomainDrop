from torch import optim


def get_optim_and_scheduler(model, network, epochs, lr, train_all=True, nesterov=False):
    if train_all:
        params = model.parameters()
    else:
        params = model.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_style(style_net, epochs, lr, nesterov=False, step_radio=0.8):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * step_radio)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_layer_joint(style_net, epochs, lr, train_all=None, nesterov=False):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * 1.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_model_lr(name, model, fc_weight=1.0):
    if 'resnet' in name:
        return [
            (model.conv1, 1.0),     # 0
            (model.bn1, 1.0),       # 1
            (model.layer1, 1.0),    # 2
            (model.domain_discriminators[1], 1.0),   # 3
            (model.layer2, 1.0),    # 4
            (model.domain_discriminators[2], 1.0),   # 5
            (model.layer3, 1.0),    # 6
            (model.domain_discriminators[3], 1.0),    # 7
            (model.layer4, 1.0),    # 8
            (model.domain_discriminators[4], 1.0),    # 9
            (model.classifier, 1.0 * fc_weight)   # 10
        ]
    elif name == 'alexnet':
        return [
            (model.layer0, 1.0),  # 0
            (model.layer1, 1.0),  # 1
            (model.layer2, 1.0),  # 2
            (model.feature_layers, 1.0),  # 3
            (model.fc, 1.0 * fc_weight),  # 4
        ]
    else:
        raise NotImplementedError


def get_optimizer(model, init_lr, momentum=.9, weight_decay=.0005, nesterov=False):
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay,
                          nesterov=nesterov)
    return optimizer


def get_optim_and_scheduler_scatter(model, network, epochs, lr, momentum=.9, weight_decay=.0005, nesterov=False, step_radio=0.8):
    model_lr = get_model_lr(name=network, model=model, fc_weight=1.0)
    optimizers = [get_optimizer(model_part, lr * alpha, momentum, weight_decay, nesterov)
                  for model_part, alpha in model_lr]
    step_size = int(epochs * step_radio)
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=step_size) for opt in optimizers]
    return optimizers, schedulers
