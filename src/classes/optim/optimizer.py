import torch


def generate_optimization_elements(config):
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
