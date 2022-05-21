import json
from glob import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd

import hydra
from hydra.utils import to_absolute_path, instantiate, call
from omegaconf import DictConfig, ListConfig

import uproot

@dataclass
class WPMaker:
    wp_definitions: dict
    tpr_step: int = 0.0001
    require_wp_vs_others: bool = True
    _taus: pd.DataFrame = None
    __converged: bool = False

    def __post_init__(self):
        self._reset_thrs()
        self.tpr = np.arange(0, 1+self.tpr_step, self.tpr_step)
        self.vs_types = self.wp_definitions.keys()

    @staticmethod
    def tau_vs_other(prob_tau, prob_other):
        return np.where(prob_tau > 0, prob_tau / (prob_tau + prob_other), np.zeros(prob_tau.shape))

    def _reset_thrs(self):
        for wp_def in self.wp_definitions.values():
            wp_def['thrs'] = []
        self.__converged = False

    def apply_wp_vs_others(self, current_vs_type):
        df = self._taus
        for other_vs_type in self.vs_types:
            if other_vs_type == current_vs_type: continue
            thrs = self.wp_definitions[other_vs_type]['thrs']
            if len(thrs) > 0:
                sel_thr = thrs[-1][-1] # select the last computed threshold for the loosest WP 
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
            thrs[vs_type] = np.quantile(taus[f'score_vs_{vs_type}'], 1 - self.tpr)
        
        # then update them in the class 
        for vs_type in self.vs_types:
            new_thrs = []
            wp_def = self.wp_definitions[vs_type]
            for eff in wp_def['wp_eff']:
                idx = (self.tpr >= eff).argmax()
                new_thrs.append(thrs[vs_type][idx])
            wp_def['thrs'].append(new_thrs)

    def print_wp(self):
        print("working_points = {")
        for vs_type in self.vs_types:
            wp_names = self.wp_definitions[vs_type]['wp_names']
            wp_thrs = self.wp_definitions[vs_type]['thrs'][-1]
            print('    "{}": {{'.format(vs_type))
            for n in reversed(range(len(wp_names))):
                name = wp_names[n]
                thr = wp_thrs[n]
                print( '        "{}": {:.7f},'.format(name, thr))
            print('    },')
        print('}')

    def run(self, n_iterations=100, verbose=False):
        if self.__converged:
            self._reset_thrs()
        
        for i in range(n_iterations):
            print("\n-> Iteration", i)
            self.update_thrs()
            if verbose:
                self.print_wp()

            # check convergence condition
            all_converged_ = True
            for wp_def in self.wp_definitions.values():
                all_converged_ = all_converged_ and len(wp_def['thrs']) >= 2 \
                    and abs(wp_def['thrs'][-1][-1] - wp_def['thrs'][-2][-1]) < 1e-7
            
            if all_converged_: 
                self.__converged = True
                print('\n-> Converged!')
                break

def create_df(path_to_preds, pred_samples, input_branches, input_tree_name, selection, tau_types):
    df = []
    path_to_preds = to_absolute_path(path_to_preds)

    # loop over input samples
    for sample_name, filename_pattern in pred_samples.items():
        json_filemap_name = f'{path_to_preds}/{sample_name}/pred_input_filemap.json'
        with open(json_filemap_name, 'r') as json_file:
            target_input_map = json.load(json_file) 

        # make list of files with predictions depending on the specified format
        if isinstance(filename_pattern, ListConfig): 
            pred_files = [f'{path_to_preds}/{sample_name}/{filename}' for filename in filename_pattern]
        elif isinstance(filename_pattern, str):
            pred_files = glob(f'{path_to_preds}/{sample_name}/{filename_pattern}')
        else:
            raise Exception(f"unknown type of filename_pattern: {type(filename_pattern)}")
        
        for pred_file in pred_files:
            # read predictions and labels
            l_ = []
            for group in ['predictions', 'targets']:
                df_ = pd.read_hdf(pred_file, group)
                df_ = df_.rename(columns={column: f'{group}_{column}' for column in df_.columns})
                l_.append(df_)
            df_pred = pd.concat(l_, axis=1)

            # read input_branches from the corresponding input file
            with uproot.open(target_input_map[pred_file]) as f:
                df_input = f[input_tree_name].arrays(input_branches, library='pd')
            
            # concatenate input branches and predictions/labels
            assert df_pred.shape[0] == df_input.shape[0], "Sizes of prediction and input dataframes don't match."
            df_ = pd.concat([df_pred, df_input], axis=1)
            assert not any(df_.isna().any(axis=0)), 'found NaNs!'
            df.append(df_)

    # select+combine taus across input samples and apply selection
    df = pd.concat(df, axis=0)
    taus = df.query(f'targets_node_tau==1')
    if selection is not None:
        taus = taus.query(selection)

    # compute vs_type discriminator scores
    for t in tau_types:
        if t != 'tau': 
            taus['score_vs_' + t] = WPMaker.tau_vs_other(taus['predictions_node_tau'].values, taus['predictions_node_' + t].values)
   
    print(f'\n-> Selected {taus.shape[0]} taus\n')
    return taus


@hydra.main(config_path='configs', config_name='derive_wp')
def main(cfg: DictConfig) -> None:
    wp_maker = instantiate(cfg.wp_maker)
    wp_maker._taus = call(cfg.create_df)
    wp_maker.run(verbose=False)
    print()
    wp_maker.print_wp()

if __name__ == '__main__':
    main()