import os
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

import mlflow
import hydra
from hydra.utils import to_absolute_path, instantiate, call
from omegaconf import DictConfig

from utils.data_processing import create_df

@dataclass
class WPMaker:
    wp_definitions: dict
    tpr_step: int = 0.0001
    require_wp_vs_others: bool = True
    WPs_to_require: dict = None
    epsilon: float = 1e-7
    n_iterations: int = 100
    verbose: bool = False
    _taus: pd.DataFrame = None
    __converged: bool = False

    def __post_init__(self):
        self.wp_definitions = {vs_type: {name: {'eff': eff, 'thrs': []} for name, eff in WPs.items()} for vs_type, WPs in self.wp_definitions.items()}
        self.tpr = np.arange(0, 1+self.tpr_step, self.tpr_step)
        self.vs_types = self.wp_definitions.keys()
        self._reset_thrs()

    def _reset_thrs(self):
        for vs_type in self.vs_types:
            WPs = self.wp_definitions[vs_type]
            for wp_cfg in WPs.values(): # for each WP per vs_type
                wp_cfg['thrs'] = []
        self.__converged = False

    def is_converged(self):
        return self.__converged

    def apply_wp_vs_others(self, current_vs_type):
        df = self._taus
        for other_vs_type in self.vs_types:
            if other_vs_type == current_vs_type: continue
            wp_to_require = self.WPs_to_require[other_vs_type]
            thrs = self.wp_definitions[other_vs_type][wp_to_require]['thrs'] # select thresholds for specified WP
            if len(thrs) > 0:
                sel_thr = thrs[-1] # select the last computed threshold
                df = df.query(f'score_vs_{other_vs_type} > {sel_thr}', inplace=False) # require passing it
        return df

    def update_thrs(self):
        thrs = {}
        # firstly compute and collect thresholds
        for vs_type in self.vs_types:
            if self.require_wp_vs_others: # apply loosest WP from the previous iteration 
                taus = self.apply_wp_vs_others(vs_type)
            else:
                taus = self._taus 
            weighted_score = DescrStatsW(data=np.array(taus[f'score_vs_{vs_type}'], dtype=np.float32), weights=np.array(taus['weight'], dtype=np.float32))
            thrs[vs_type] = weighted_score.quantile(probs=1-self.tpr, return_pandas=False)

        # then update them in the class 
        for vs_type, WPs in self.wp_definitions.items():
            for wp_cfg in WPs.values():
                idx = (self.tpr >= wp_cfg["eff"]).argmax()
                wp_cfg['thrs'].append(thrs[vs_type][idx])

    def print_wp(self):
        print("\nworking_points = {")
        for vs_type, WPs in self.wp_definitions.items():
            print('    "{}": {{'.format(vs_type))
            for wp_name, wp_cfg in reversed(WPs.items()):
                print( '        "{}": {:.4f},'.format(wp_name, wp_cfg['thrs'][-1]))
            print('    },')
        print('}')

    def run(self):
        if self.__converged:
            self._reset_thrs()
        
        for i in range(self.n_iterations):
            print("\n-> Iteration", i)
            self.update_thrs()
            if self.verbose:
                self.print_wp()

            # check convergence condition
            all_converged = True
            for WPs in self.wp_definitions.values():
                for wp_cfg in WPs.values():
                    if len(wp_cfg['thrs']) >= 2:
                        all_converged = all_converged and abs(wp_cfg['thrs'][-1] - wp_cfg['thrs'][-2]) < self.epsilon
                    else: 
                        all_converged = False

            if all_converged: 
                self.__converged = True
                print('\n-> Converged!')
                break

@hydra.main(config_path='configs', config_name='derive_wp')
def main(cfg: DictConfig) -> None:
    wp_maker = instantiate(cfg['wp_maker'])
    wp_maker._taus = call(cfg['create_df'])
    wp_maker.run()
    wp_maker.print_wp()

    if wp_maker.is_converged():
        if os.path.exists(path_to_mlflow:=to_absolute_path(cfg['create_df']['path_to_mlflow'])):
            mlflow.set_tracking_uri(f"file://{path_to_mlflow}")
            if mlflow.get_experiment(experiment_id:=str(cfg['create_df']['experiment_id'])) and mlflow.get_run(run_id:=cfg['create_df']['run_id']):
                path_to_artifacts = f'{path_to_mlflow}/{experiment_id}/{run_id}/artifacts'
                
                # store threshold values for all iterations
                with open(f'{path_to_artifacts}/working_points.log', 'w') as json_file:
                    json_file.write(json.dumps(wp_maker.wp_definitions, indent=4))
                
                # store final thresholds
                wp_definitions_final = {}
                for vs_type, WPs in wp_maker.wp_definitions.items():
                    wp_definitions_final[vs_type] = {wp_name: round(wp_cfg['thrs'][-1], 4) for wp_name, wp_cfg in reversed(WPs.items())}

                with open(f'{path_to_artifacts}/working_points.json', 'w') as json_file:
                    json_file.write(json.dumps(wp_definitions_final, indent=4))

                print(f'\n[INFO] logged working points to artifacts for experiment ({experiment_id}), run ({run_id})\n')
    else:
        print('\nWPMaker haven\'t converged!\n')

if __name__ == '__main__':
    main()