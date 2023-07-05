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

    if config.task in ["segmentation", "classification", "change-detection", "detection"]:
        if config.task == "segmentation":
            optimizer = torch.optim.SGD
            optimizer_params = {
                "lr": config.lr,
                "momentum": config.momentum,
            }
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_params = {}
            scheduler_interval = "epoch"

        elif config.task == "classification":
            optimizer = torch.optim.SGD
            optimizer_params = {
                "lr": config.lr,
                "momentum": config.momentum,
            }
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_params = {}
            scheduler_interval = "epoch"

        elif config.task == "change-detection":
            optimizer = torch.optim.SGD
            optimizer_params = {
                "lr": config.lr,
                "momentum": config.momentum,
            }
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_params = {}
            scheduler_interval = "epoch"

        elif config.task == "detection":
            optimizer = torch.optim.SGD
            optimizer_params = {
                "lr": config.lr,
                "momentum": config.momentum,
            }
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_params = {"monitor": config.monitor, "mode": config.mode}
            scheduler_interval = "epoch"

        return (
            optimizer,
            optimizer_params,
            scheduler,
            scheduler_params,
            scheduler_interval,
        )

    else:
        print("La tâche demandée n'est pas reconnue")
