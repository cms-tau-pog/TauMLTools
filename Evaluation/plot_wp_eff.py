import json
import pandas as pd
import mlflow
import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import OmegaConf, DictConfig

from utils.data_processing import create_df
from utils.wp_eff import differential_efficiency, plot_efficiency

@hydra.main(config_path='configs', config_name='plot_wp_eff')
def main(cfg: DictConfig) -> None:
    
    # read WP thresholds from mlflow artifacts
    path_to_mlflow = to_absolute_path(cfg['create_df']['path_to_mlflow'])
    mlflow.set_tracking_uri(f"file://{path_to_mlflow}")
    experiment_id = cfg['create_df']['experiment_id']
    run_id = cfg['create_df']['run_id']
    path_to_wp_definitions = f"{path_to_mlflow}/{experiment_id}/{run_id}/artifacts/working_points.json"
    with open(path_to_wp_definitions, 'r') as f:
        wp_definitions = json.load(f)

    # instantiate partial create_df object from hydra cfg
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    partial_create_df = instantiate(cfg["create_df"])
    partial_plot_efficiency = instantiate(cfg["var_cfg"])
    
    # make dataframes with specified input features and predictions for each tau_type
    vs_type = cfg['vs_type']
    df_sel = {}
    if not cfg['from_skims']: # create dataframes and log to mlflow 
        with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as active_run:
            for tau_type in ['tau', vs_type]:
                df = partial_create_df(tau_type_to_select=tau_type, pred_samples=cfg['pred_samples'][tau_type])
                df.to_csv(f'{tau_type}.csv')
                mlflow.log_artifact(f'{tau_type}.csv', cfg['output_skim_folder'])
                df_sel[tau_type] = df
    else: # read already existing skimmed dataframes
        for tau_type in ['tau', vs_type]:
            df = pd.read_csv(f"{path_to_mlflow}/{experiment_id}/{run_id}/artifacts/{cfg['output_skim_folder']}/{tau_type}.csv")
            df_sel[tau_type] = df

    # compute and plot efficiency curves
    wp_thrs, wp_names = list(wp_definitions[vs_type].values()), list(wp_definitions[vs_type].keys())
    WPs_to_require = OmegaConf.to_object(cfg['WPs_to_require'])
    del WPs_to_require[vs_type] # remove vs_type key
    eff, eff_up, eff_down = differential_efficiency(df_sel['tau'], df_sel[vs_type],
                                                    cfg['var_cfg']['var_name'], cfg['var_cfg']['var_bins'], 
                                                    vs_type, 'score_vs_', wp_thrs,
                                                    cfg['require_WPs_in_numerator'], cfg['require_WPs_in_denominator'],
                                                    WPs_to_require, wp_definitions)
    fig = partial_plot_efficiency(eff=eff, eff_up=eff_up, eff_down=eff_down, labels=wp_names)

    # log to mlflow
    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as active_run:
        mlflow.log_figure(fig, cfg['output_filename'])
    print(f'\n[INFO] logged plots to artifacts for experiment ({experiment_id}), run ({run_id})\n')

if __name__ == '__main__':
    main()