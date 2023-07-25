import torch


def generate_optimization_elements(config, optimizer_dict, scheduler_dict, nb_total_images):
    """
    Returns the optimization elements required for PyTorch training.

    Args:
        config (dict): The configuration dictionary
        containing the optimization parameters.

    Returns:
        tuple: A tuple containing the optimizer, optimizer parameters,
        scheduler, scheduler parameters, and scheduler interval.

    """
    task_liste = ["segmentation", "classification", "change-detection"]
    task = config["donnees"]["task"]

    if task in task_liste:
        if ((1 - config["optim"]["val prop"])*nb_total_images)%config["optim"]["batch size"] == 0:
            steps_per_epoch = int(((1 - config["optim"]["val prop"])*nb_total_images)/config["optim"]["batch size"])
        else:
            steps_per_epoch = int(((1 - config["optim"]["val prop"])*nb_total_images)/config["optim"]["batch size"]) + 1

        optimizer = optimizer_dict[config["optim"]["optimizer"]]
        scheduler = scheduler_dict[config["optim"]["scheduler"]]

        if config["optim"]["optimizer"] == "SGD":
            optimizer_params = {
                "lr": config["optim"]["lr"],
                "momentum": config["optim"]["momentum"]}
        elif config["optim"]["optimizer"] == "Adam":
            optimizer_params = {
                "lr": config["optim"]["lr"]}
        
        if config["optim"]["scheduler"] == "ReduceLROnPlateau":
            scheduler_params = {}
        elif config["optim"]["scheduler"] == "OneCycleLR":
            scheduler_params = {
                "max_lr": 0.005,
                "epochs": config["optim"]["max epochs"],
                "steps_per_epoch": steps_per_epoch}
            
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
