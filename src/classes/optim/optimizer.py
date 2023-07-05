import torch


def generate_optimization_elements(config):
    """
    Returns the optimization elements required for PyTorch training.

    Args:
        config (dict): The configuration dictionary
        containing the optimization parameters.

    Returns:
        tuple: A tuple containing the optimizer, optimizer parameters,
        scheduler, scheduler parameters, and scheduler interval.

    """

    optimizer = torch.optim.SGD
    optimizer_params = {
        "lr": config.lr,
        "momentum": config.momentum,
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        "monitor": config.earlystop["monitor"],
        "mode": config.earlystop["mode"],
    }  # TODO: vérifiersi ok d'utilise config d'early stop ici
    scheduler_interval = "epoch"

    return (
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval,
    )
