import mlflow

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

@hydra.main(config_path='.', config_name='log_to_mlflow')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None: # fetch existing experiment id
        run_kwargs = {'experiment_id': experiment.experiment_id} 
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        print(f'\nCreated experiment {cfg.experiment_name} with ID: {experiment_id}')
        run_kwargs = {'experiment_id': experiment_id} 
    if cfg.run_id is not None:
        run_kwargs['run_id'] = cfg.run_id # to log data into existing run

    with mlflow.start_run(**run_kwargs) as run:
        mlflow.log_param('run_id', run.info.run_id)
        if cfg.files_to_log is not None:
            for path_to_file, dir_to_log_to in cfg.files_to_log.items():
                print(f'-> logging {path_to_file}')
                mlflow.log_artifact(to_absolute_path(path_to_file), dir_to_log_to)
        if cfg.params_to_log is not None:
            mlflow.log_params(cfg.params_to_log)
        print(f'\nLogged to run ID: {run.info.run_id}\n')

if __name__ == '__main__':
    main()