import json
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
    vs_types = ['e', 'mu', 'jet']
    vs_type = cfg['vs_type']
    df_sel = {}
    for tau_type in ['tau', vs_type]:
        df = partial_create_df(tau_type_to_select=tau_type, pred_samples=cfg['pred_samples'][tau_type])
        if cfg['require_wp_vs_others']:
            for other_vs_type in vs_types:
                if other_vs_type == vs_type: continue
                wp_to_require = cfg['WPs_to_require'][other_vs_type]
                thr = wp_definitions[other_vs_type][wp_to_require] # select thresholds for specified WP
                df = df.query(f'score_vs_{other_vs_type} > {thr}', inplace=False) # require passing it
                print(f'   After passing required {wp_to_require} WP vs. {other_vs_type}: {df.shape[0]}')
        df_sel[tau_type] = df
        print()

    # compute and plot efficiency curves
    wp_thrs, wp_names = list(wp_definitions[vs_type].values()), list(wp_definitions[vs_type].keys())
    eff, eff_up, eff_down = differential_efficiency(df_sel['tau'], df_sel[vs_type], 
                                                             cfg['var_cfg']['var_name'], cfg['var_cfg']['var_bins'],
                                                            'score_vs_' + vs_type, wp_thrs)
    fig = partial_plot_efficiency(eff=eff, eff_up=eff_up, eff_down=eff_down, labels=wp_names)

    # log to mlflow
    mlflow.set_tracking_uri(f"file://{path_to_mlflow}")
    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as active_run:
        mlflow.log_figure(fig, cfg['output_filename'])
    print(f'\n[INFO] logged plots to artifacts for experiment ({experiment_id}), run ({run_id})\n')

if __name__ == '__main__':
    main()