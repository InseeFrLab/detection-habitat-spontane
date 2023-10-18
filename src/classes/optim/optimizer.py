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

    optimizer = torch.optim.Adam if config.task == "detection" else torch.optim.SGD
    optimizer_params = {
        "lr": config.lr,
        "momentum": config.momentum,
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        "monitor": config.earlystop["monitor"],
        "mode": config.earlystop["mode"],
        "patience": config.scheduler_patience,
    }  # TODO: vérifier si ok d'utilise config d'early stop ici.
    # IMPORTANT CAR PEUT ETRE CONFIG A REVOIR
    scheduler_interval = "epoch"

    return (
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval,
    )
