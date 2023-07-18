import gc
import os
import sys

import mlflow
import torch

from configurators.configurator import Configurator
from dico_config import task_to_evaluation
from instantiators.instantiator import Instantiator
from preprocessors.preprocessor import Preprocessor
from utils.utils import get_root_path, update_storage_access


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

    # TODO :  Download data devrait rien retourner donc à améliorer
    preprocessor.download_data()
    preprocessor.prepare_train_data(configurator)
    preprocessor.prepare_test_data(configurator)

    train_dl, valid_dl, test_dl = instantiator.dataloader(configurator)
    trainer = instantiator.trainer()

    light_module = instantiator.lightning_module()

    torch.cuda.empty_cache()
    gc.collect()

    if configurator.mlflow:
        update_storage_access()
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
        mlflow.end_run()
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(experiment_name)
        # mlflow.pytorch.autolog()

        with mlflow.start_run(run_name=run_name):
            mlflow.autolog()
            mlflow.log_artifact(get_root_path() / "config.yml", artifact_path="config.yml")
            trainer.fit(light_module, train_dl, valid_dl)

            if configurator.source_train == "PLEIADES":
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

                model = light_module_checkpoint.model
                try:
                    print(model.device)
                except Exception:
                    pass

                if configurator.task not in task_to_evaluation:
                    raise ValueError("Invalid task type")
                else:
                    evaluer_modele_sur_jeu_de_test = task_to_evaluation[configurator.task]

                evaluer_modele_sur_jeu_de_test(
                    test_dl,
                    model,
                    configurator.tile_size,
                    configurator.batch_size_test,
                    configurator.n_bands,
                    configurator.mlflow,
                    device,
                )

    else:
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
        model = light_module_checkpoint.model

        if configurator.src_task not in task_to_evaluation:
            raise ValueError("Invalid task type")
        else:
            evaluer_modele_sur_jeu_de_test = task_to_evaluation[configurator.src_task]

        evaluer_modele_sur_jeu_de_test(
            test_dl,
            model,
            configurator.tile_size,
            configurator.batch_size_test,
            configurator.n_bands,
            configurator.mlflow,
            device,
        )


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    run_pipeline(remote_server_uri, experiment_name, run_name)

# nohup python run_training_pipeline.py
