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
    # TO DO à développer selon la config ?
    optimizer = torch.optim.SGD
    optimizer_params = {
        "lr": config["optim"]["lr"],
        "momentum": config["optim"]["momentum"],
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {}
    scheduler_interval = "epoch"

    return\
        optimizer, optimizer_params,\
        scheduler,  scheduler_params,\
        scheduler_interval
