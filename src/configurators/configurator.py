"""
Configurator class.
"""
import yaml
from yaml.loader import SafeLoader


class Configurator:
    """
    Configurator class.
    """

    def __init__(self, config_path: str) -> None:
        """
        Constructor for the Configurator class.
        """
        with open(config_path) as f:
            config = yaml.load(f, Loader=SafeLoader)

        self.year = config["data"]["year"]
        self.dep = config["data"]["dep"]
        self.type_labeler = config["data"]["type_labeler"]
        self.buffer_size = config["data"]["buffer_size"]
        self.dataset = config["data"]["dataset"]
        self.dataset_test = config["data"]["dataset_test"]
        self.n_bands = config["data"]["n_bands"]
        self.task = config["data"]["task"]
        self.prop = config["data"]["prop"]
        self.source_train = config["data"]["source_train"]
        self.augmentation = config["data"]["augmentation"]
        self.tile_size = config["data"]["tile_size"]
        self.n_channels_train = config["data"]["n_channels_train"]
        self.num_workers = config["data"]["num_workers"]
        self.src_task = f"{self.source_train}{self.task}"

        self.loss = config["optim"]["loss"]
        self.lr = config["optim"]["lr"]
        self.momentum = config["optim"]["momentum"]
        self.batch_size = config["optim"]["batch_size"]
        self.batch_size_test = config["optim"]["batch_size_test"]
        self.max_epochs = config["optim"]["max_epochs"]
        self.module = config["optim"]["module"]
        self.val_prop = config["optim"]["val_prop"]
        self.accumulate_batch = config["optim"]["accumulate_batch"]
        self.monitor = config["optim"]["monitor"]
        self.mode = config["optim"]["mode"]
        self.patience = config["optim"]["patience"]

        self.mlflow = config["mlflow"]
