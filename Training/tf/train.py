import shutil
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import tensorflow_addons as tfa

from models.taco import TacoNet
from models.transformer import Transformer, CustomSchedule
from models.particle_net import ParticleNet
from utils.training import compose_datasets, log_to_mlflow

import mlflow
mlflow.tensorflow.autolog(log_models=False) 

@hydra.main(config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:

    # setup gpu
    physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[cfg["gpu_id"]], True)
    tf.config.set_logical_device_configuration(
            physical_devices[cfg["gpu_id"]],
            [tf.config.LogicalDeviceConfiguration(memory_limit=cfg["memory_limit"]*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    # set up mlflow experiment id
    mlflow.set_tracking_uri(f'file://{to_absolute_path(cfg["path_to_mlflow"])}')
    experiment = mlflow.get_experiment_by_name(cfg["experiment_name"])
    if experiment is not None: # fetch existing experiment id
        run_kwargs = {'experiment_id': experiment.experiment_id}
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg["experiment_name"])
        run_kwargs = {'experiment_id': experiment_id}

    # start mlflow run
    with mlflow.start_run(**run_kwargs) as active_run:
        run_id = active_run.info.run_id
        
        # load datasets 
        train_data, val_data = compose_datasets(cfg["datasets"], cfg["tf_dataset_cfg"])

        # define model
        feature_name_to_idx = {}
        for particle_type, names in cfg["feature_names"].items():
            feature_name_to_idx[particle_type] = {name: i for i, name in enumerate(names)}
        if cfg["model"]["type"] == 'taco_net':
            model = TacoNet(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"])
        elif cfg["model"]["type"] == 'transformer':
            model = Transformer(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"])
        elif cfg['model']['type'] == 'particle_net':
            model = ParticleNet(feature_name_to_idx, cfg['model']['kwargs']['encoder'], cfg['model']['kwargs']['decoder'])
        else:
            raise RuntimeError('Failed to infer model type')
        X_, _ = next(iter(train_data))
        model(X_) # init it for correct autologging with mlflow

        # LR schedule
        if cfg['schedule'] is None: 
            learning_rate = cfg["learning_rate"]
        elif cfg['schedule']=='custom':
            learning_rate = CustomSchedule(float(cfg["model"]["kwargs"]["encoder"]["dim_model"]), float(cfg['warmup_steps']), float(cfg['lr_multiplier']))
        elif cfg['schedule']=='decrease':
            def scheduler(epoch, lr):
                if epoch%cfg['decrease_every']!=0 or epoch==0:
                    return lr
                else:
                    return lr / cfg['decrease_by']
            learning_rate = cfg["learning_rate"]
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        else:
            raise RuntimeError(f"Unknown value for schedule: {cfg['schedule']}. Only \'custom\', \'decrease\' and \'null\' are supported.")

        # optimiser
        if cfg['optimiser']=='adam': 
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
        elif cfg['optimiser']=='sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cfg['momentum'], nesterov=cfg['nesterov'])
        elif cfg['optimiser']=='adamw':
            opt = tfa.optimizers.AdamW(weight_decay=cfg['weight_decay'], learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
        elif cfg['optimiser']=='radam':
            opt = tfa.optimizers.RectifiedAdam(weight_decay=cfg['weight_decay'], learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
        else:
            raise RuntimeError(f"Unknown value for optimiser: {cfg['optimiser']}. Only \'sgd\' and \'adam\' are supported.")

        # callbacks, compile, fit
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=cfg["min_delta"], patience=cfg["patience"], mode='auto', restore_best_weights=True)
        checkpoint_path = 'tmp_checkpoints'
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + "/" + "epoch_{epoch:02d}---val_loss_{val_loss:.3f}",
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_freq='epoch',
            save_best_only=False)
        callbacks = [early_stopping, model_checkpoint] if cfg['schedule']!='descrease' else [early_stopping, model_checkpoint, lr_scheduler]
        model.compile(optimizer=opt,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                    metrics=['accuracy', tf.keras.metrics.AUC(from_logits=False)])
        model.fit(train_data, validation_data=val_data, epochs=cfg["n_epochs"], callbacks=callbacks, verbose=1)  #  steps_per_epoch=1000, 

        # save model
        print("\n-> Saving model")
        model.save((f'{cfg["model"]["name"]}.tf'), save_format="tf")
        mlflow.log_artifacts(f'{cfg["model"]["name"]}.tf', 'model')
        if cfg["model"]["type"] == 'taco_net':
            print(model.wave_encoder.summary())
            print(model.wave_decoder.summary())
        elif cfg["model"]["type"] == 'transformer':
            print(model.summary())
        elif cfg['model']['type'] == 'particle_net':
            print(model.summary())

        # log info
        log_to_mlflow(model, cfg)
        mlflow.log_param('run_id', run_id)
        mlflow.log_artifacts(checkpoint_path, "checkpoints")
        shutil.rmtree(checkpoint_path)

        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg["experiment_name"]}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')

if __name__ == '__main__':
    main()