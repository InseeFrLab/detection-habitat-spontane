import gc
import os
import sys

import mlflow
import torch

from configurators.configurator import Configurator
from evaluators.evaluator import Evaluator
from instantiators.instantiator import Instantiator
from preprocessors.preprocessor import Preprocessor
from utils.utils import get_root_path


def run_pipeline(remote_server_uri, experiment_name, run_name):
    """
    Runs the segmentation pipeline u
    sing the configuration specified in `config.yml`
    and the provided MLFlow parameters.
    Args:
        None
    Returns:
        None
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Open the file and load the file
    configurator = Configurator(get_root_path() / "config.yml", get_root_path() / "environment.yml")

    instantiator = Instantiator(configurator)
    preprocessor = Preprocessor(configurator)
    evaluator = Evaluator(configurator)

    preprocessor.download_data()
    preprocessor.prepare_train_data()
    preprocessor.prepare_test_data()

    train_dl, valid_dl, test_dl = instantiator.dataloader()
    trainer = instantiator.trainer()

    light_module = instantiator.lightning_module()

    torch.cuda.empty_cache()
    gc.collect()

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
    mlflow.end_run()
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.autolog()
        mlflow.log_artifact(get_root_path() / "config.yml", artifact_path="config.yml")
        trainer.fit(light_module, train_dl, valid_dl)

        light_module_checkpoint = light_module.load_from_checkpoint(
            loss=instantiator.loss(),
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            model=light_module.model,
            optimizer=light_module.optimizer,
            optimizer_params=light_module.optimizer_params,
            scheduler=light_module.scheduler,
            scheduler_params=light_module.scheduler_params,
            scheduler_interval=light_module.scheduler_interval,
        )

        evaluator.evaluate_model(test_dl, light_module_checkpoint.model, device)


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    run_pipeline(remote_server_uri, experiment_name, run_name)
