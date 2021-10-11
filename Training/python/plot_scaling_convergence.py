import uproot

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import json
import yaml
from glob import glob
import os
import sys
from collections import defaultdict
import click

sns.set_theme(context='notebook', font='sans-serif', style='white', palette=None, font_scale=1.5,
              rc={"lines.linewidth": 2.5, "font.sans-serif": 'DejaVu Sans', "text.usetex": True})

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_params_to_df(file_names, cone_type, normal_features):
    mean_dict, std_dict = defaultdict(list), defaultdict(list)
    for file_name in file_names:
        with open(file_name) as f:
            scalings = json.load(f)
        for var_type in scalings.keys():
            mean, std = {}, {}
            for feature, feature_scaling in scalings[var_type].items():
                if cone_type not in feature_scaling: continue
                if feature not in normal_features[var_type]: continue
                mean[feature] = feature_scaling[cone_type]['mean']
                std[feature] = feature_scaling[cone_type]['std']
            mean_dict[var_type].append(mean)
            std_dict[var_type].append(std)
    df_mean_dict = {var_type: pd.DataFrame(mean_dict[var_type]) for var_type in mean_dict}
    df_std_dict = {var_type: pd.DataFrame(std_dict[var_type]) for var_type in std_dict}
    return df_mean_dict, df_std_dict

def plot_running_diff(running_diff_mean, running_diff_std,
                      n_files_per_step, var_type, cone_type, savepath=None, close_plot=False):
    fig, axs = plt.subplots(2, 1, figsize=(23,8))

    assert len(running_diff_mean)==len(running_diff_std)
    n_snapshots = len(running_diff_mean)
    axs[0].boxplot(list(running_diff_mean.values), notch=True, bootstrap=10000, whis=(5, 95))
    # axs[0].set_title(f'\text{var_type}{cone_type}')
    axs[0].set_ylabel(r'$\frac{\Big|\mathrm{mean}[i] - \mathrm{mean}[i-1]\Big|}{\Big|\mathrm{mean}[i-1]\Big|}$')
    axs[0].set_yscale('log')
    axs[0].set_xlim(.3, axs[0].get_xlim()[1])
    axs[0].set_xticks(list(range(1, n_snapshots+1)))
    axs[0].set_xticklabels(list(range(1, n_snapshots))+['all'])
    axs[0].hlines(.1, axs[0].get_xlim()[0], axs[0].get_xlim()[1]-1, ls=':', linewidth=3, label='10\% level')
    axs[0].vlines(49.5, *axs[0].get_ylim(), 'black', linewidth=1.5)
    axs[0].set_title(f"{var_type.replace('_', '-')}, {cone_type.replace('_', '-')}")
    #
    axs[1].boxplot(list(running_diff_std.values), notch=True, bootstrap=10000, whis=(5, 95))
    axs[1].set_ylabel(r'$\frac{\Big|\mathrm{std}[i] - \mathrm{std}[i-1]\Big|}{\Big|\mathrm{std}[i-1]\Big|}$')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(f'snapshot i, increment of {n_files_per_step} files per step')
    axs[1].set_xlim(.3, axs[1].get_xlim()[1])
    axs[1].set_xticks(list(range(1, n_snapshots+1)))
    axs[1].set_xticklabels(list(range(1, n_snapshots))+['all'])
    axs[1].hlines(.1, axs[1].get_xlim()[0], axs[1].get_xlim()[1]-1, ls=':', linewidth=3, label='10\% level')
    axs[1].vlines(49.5, *axs[1].get_ylim(), 'black', linewidth=1.5)
    # plt.legend(loc='upper center')
    plt.show()
    if savepath is not None:
        fig.savefig(f'{savepath}/{var_type}_{cone_type}.png')
    if close_plot:
        plt.close()

@click.command(
    help="Plot the convergence of scaling parameters' computation as a running difference between consequent snapshots."
)
@click.option("--train-cfg", type=str, default='../configs/training_v1.yaml', help="Path to yaml configuration file used for training", show_default=True)
@click.option('--snapshots', type=str, default='/afs/cern.ch/work/o/ofilatov/public/scaling_v1/TauFlat', help='Path to the directory with json snapshot files', show_default=True)
@click.option('-n', '--nfiles-per-step', 'N_FILES_PER_STEP', type=int, default=10, help='Number of processed files per log step, as it was set while running the scaling computation', show_default=True)
@click.option('-v', '--version', type=str, default='v1_tau', help='Postfix in the name of json files which tags a specific scaling version', show_default=True)
@click.option('-t', '--var-type', 'VAR_TYPE', type=str, default='TauFlat', help='Variables\' type for which scaling parameters will be plotted.', show_default=True)
@click.option('-c', '--cone-type', 'CONE_TYPE', type=str, default='global', help='Cone type for which scaling parameters will be plotted.', show_default=True)
@click.option('-o', "--output-folder", type=str, default='.', help='Output directory to save plots.', show_default=True)
def main(
    train_cfg, snapshots, version, N_FILES_PER_STEP,
    VAR_TYPE, CONE_TYPE, output_folder
):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # open the main training cfg
    with open(train_cfg) as f:
        training_cfg = yaml.safe_load(f)

    # scaling json doesn't store scaling_type, so read it from the scaling cfg
    normal_features = defaultdict(list) # interested only in scaling type "normal", where means/stds were computed
    for var_type, var_list in training_cfg['Features_all'].items():
        for var_dict in var_list:
            var_name = list(var_dict.keys())[0]
            if var_dict[var_name][2] == 'normal':
                normal_features[var_type].append(var_name)
    var_types = list(normal_features.keys())

    # read json file names into list
    n_snapshots = len(glob(f'{snapshots}/scaling_params_{version}_log_*.json'))
    sorted_file_names = [f'{snapshots}/scaling_params_{version}_log_{i}.json' for i in range(n_snapshots)] # sorting in ascending order
    sorted_file_names += [f'{snapshots}/scaling_params_{version}.json'] # append final snapshot to the end

    # read scaling params into pandas DataFrame
    df_mean_dict, df_std_dict = read_params_to_df(sorted_file_names, CONE_TYPE, normal_features)
    df_mean = df_mean_dict[VAR_TYPE]
    df_std = df_std_dict[VAR_TYPE]
    if len(df_mean.columns)==0 and len(df_std.columns)==0:
        raise Exception(f"found no variables with scaling type: 'normal', cone type {CONE_TYPE}: '{VAR_TYPE}'")

    # drop nans and remove inf columns
    not_inf_columns = list(df_mean.columns[~np.isinf(df_mean).any()])
    df_mean = df_mean[not_inf_columns].dropna(axis=1, how='any')
    if np.any(np.isinf(df_mean)) > 0:
        print("Found columns with inf in df_mean: ", df_mean.columns[np.isinf(df_mean)])

    not_inf_columns = list(df_std.columns[~np.isinf(df_std).any()])
    df_std = df_std[not_inf_columns].dropna(axis=1, how='any')
    if np.any(np.isinf(df_std)) > 0:
        print("Found columns with inf in df_std: ", df_std.columns[np.isinf(df_std)])

    # running diffs = difference of mean/std value between two consecutive snapshots, normalised by the value
    running_diff_mean = abs(df_mean.iloc[1:].values - (df_mean.iloc[0:-1].values)) / abs(df_mean.iloc[0:-1])
    running_diff_std = abs(df_std.iloc[1:].values - (df_std.iloc[0:-1].values)) / abs(df_std.iloc[0:-1])

    # append to the end difference between 1st and final snapshots
    running_diff_mean.iloc[-1] = abs(df_mean.iloc[-1].values - (df_mean.iloc[0].values)) / abs(df_mean.iloc[0])
    running_diff_std.iloc[-1] = abs(df_std.iloc[-1].values - (df_std.iloc[0].values)) / abs(df_std.iloc[0])
    plot_running_diff(running_diff_mean, running_diff_std, N_FILES_PER_STEP, VAR_TYPE, CONE_TYPE, savepath=output_folder, close_plot=True)

if __name__ == '__main__':
    main()
