import os
import json
import yaml
import click

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', font='sans-serif', style='white', palette=None, font_scale=1.5,
              rc={"lines.linewidth": 2.5, "font.sans-serif": 'DejaVu Sans', "text.usetex": False})

def plot_ranges(file_id, var_name, cone_type, mean, median, min_value, max_value,
                clamp_range, one_sigma_range, two_sigma_range, three_sigma_range,
                suspicious_dict=None, savepath=None, close_plot=False):
    assert len(clamp_range)==2 and clamp_range[0] <= clamp_range[1]
    assert len(one_sigma_range)==2 and one_sigma_range[0] <= one_sigma_range[1]
    assert len(two_sigma_range)==2 and two_sigma_range[0] <= two_sigma_range[1]
    assert len(three_sigma_range)==2 and three_sigma_range[0] <= three_sigma_range[1]

    ### figure configuration
    fig = plt.figure(figsize=(10.3, 5.0))
    xscale = clamp_range[1]-clamp_range[0]
    yscale = 1
    level_quantile = 0
    level_clamp = level_quantile+3*yscale
    bar_height = 1.5*yscale
    xlim = [clamp_range[0]-.5*xscale, clamp_range[1]+.5*xscale]
    ylim = [level_quantile-2*yscale, level_clamp+13.*yscale]

    ### plot range to be clamped
    if mean is not None:
        plt.vlines(mean, level_clamp-1.1*yscale, level_clamp+1.1*yscale, color='steelblue', alpha=1., label='mean')
    plt.barh(level_clamp, width=clamp_range[1]-clamp_range[0], height=bar_height, left=clamp_range[0], alpha=0.3, label='clamping range')
    plt.hlines(level_clamp, *xlim, color='black', linestyles='dotted', alpha=0.15)

    ### plot quantiles as measured in data
    plt.vlines(median, level_quantile-1.1*yscale, level_quantile+1.1*yscale, color='black', linestyles='solid', alpha=1., label='median')
    plt.hlines(level_quantile, *xlim, color='black', linestyles='dotted', alpha=0.15)
    plt.barh(level_quantile, width=one_sigma_range[1]-one_sigma_range[0], height=bar_height, left=one_sigma_range[0], color='black', alpha=0.45, label=f'1 sigma (IQW={norm.cdf(1)-norm.cdf(-1):.2f})')
    plt.barh(level_quantile, width=two_sigma_range[1]-two_sigma_range[0], height=bar_height, left=two_sigma_range[0], color='black', alpha=0.3, label=f'2 sigma (IQW={norm.cdf(2)-norm.cdf(-2):.2f})')
    plt.barh(level_quantile, width=three_sigma_range[1]-three_sigma_range[0], height=bar_height, left=three_sigma_range[0], color='black', alpha=0.15, label=f'3 sigma (IQW={norm.cdf(3)-norm.cdf(-3):.3f})')

    ### add mean/median/min/max numbers and suspicious tag to the plot
    if mean is not None:
        plt.text(xlim[0] + 0.05*(xlim[1]-xlim[0]), ylim[1]-0.15*(ylim[1]-ylim[0]), f'mean: {mean:.1e}') # , color='steelblue'
    plt.text(xlim[0] + 0.05*(xlim[1]-xlim[0]), ylim[1]-0.3*(ylim[1]-ylim[0]), f'median: {median:.1e}\nmin, max: ({min_value:.1e}, {max_value:.1e})')
    if suspicious_dict is not None:
        plt.text(xlim[0] + 0.05*(xlim[1]-xlim[0]), ylim[1]-0.55*(ylim[1]-ylim[0]), '\n'.join([k for k, v in suspicious_dict.items() if v]), color='lightcoral')

    ### plot and save
    plt.title(f'{var_name}, {cone_type} [fid: {file_id}]')
    plt.yticks(ticks=[level_clamp, level_quantile], labels=['clamp', 'quantile'])
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    if not close_plot: plt.show()
    if savepath is not None:
        fig.savefig(f'{savepath}/{var_name}_{cone_type}.png')
    if close_plot:
        plt.close()

@click.command(
    help="Plot for input features their interquantile ranges along with ranges to be clamped as a part of scaling step."
)
@click.option("--train-cfg", type=str, default='../configs/training_v1.yaml', help="Path to yaml configuration file used for training", show_default=True)
@click.option("--scaling-file", type=str, help="Path to json file with scaling parameters")
@click.option("--quantile-file", type=str, help="Path to json file with quantile parameters")
@click.option("--file-id", type=int, default=0, help="File ID to be picked from quantile parameters file", show_default=True)
@click.option("--output-folder", type=str, default='scaling_plots/quantiles', help="Folder to store range plots", show_default=True)
@click.option('--only-suspicious', type=bool, default=False, show_default=True )
def main(
    train_cfg, scaling_file, quantile_file, file_id,
    output_folder, only_suspicious
):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    with open(train_cfg) as f:
        training_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(scaling_file) as f:
        scaling_params = yaml.load(f, Loader=yaml.FullLoader)
    with open(quantile_file) as f:
        quantile_params = yaml.load(f, Loader=yaml.FullLoader)

    ### fetch type of scaling for a given variable
    for var_type in training_cfg['Features_all']: # loop over types of variables (particle types) specified in the training cfg
        if not os.path.isdir(f'{output_folder}/{var_type}'):
            os.makedirs(f'{output_folder}/{var_type}')
        print('\n\n')
        print(f'  <{var_type}>')
        print()
        for var_dict in training_cfg['Features_all'][var_type]: # loop over variables therein
            assert len(var_dict.keys()) == 1
            var_name = list(var_dict.keys())[0]
            var_scaling_type = var_dict[var_name][2]
            if var_scaling_type=='no_scaling' or var_scaling_type=='categorical':
                continue
            if var_type not in scaling_params:
                print(f'[INFO] Variable type ({var_type}) is not present in the scaling json file, skipping it.')
                continue
            if var_name not in scaling_params[var_type]:
                print(f'[INFO] Variable ({var_name}) is not present for variable type ({var_type}) in the scaling json file, skipping it.')
                continue

            # loop over cone types for which scaling params were computed
            for cone_type in scaling_params[var_type][var_name]:
                ### fetch variable's quantile and scaling dictionaries
                try:
                    var_quantiles = quantile_params[var_type][var_name][cone_type][str(file_id)]
                except:
                    print(f'[INFO] Failed to retrieve quantile parameters for var_name={var_name} and cone_type={cone_type}: skipping this variable')
                    continue
                try:
                    var_scaling = scaling_params[var_type][var_name][cone_type]
                except:
                    print(f'[INFO] Failed to retrieve scaling parameters for var_name={var_name} and cone_type={cone_type}: skipping this variable')
                    continue

                ### fetch mean for clamping params
                mean = var_scaling['mean']

                ### fetch quantiles
                median = var_quantiles['median']
                min_value = var_quantiles['min']
                max_value = var_quantiles['max']
                #
                one_sigma_left = var_quantiles['1sigma']['left']
                one_sigma_right = var_quantiles['1sigma']['right']
                one_sigma_range = [one_sigma_left, one_sigma_right]
                #
                two_sigma_left = var_quantiles['2sigma']['left']
                two_sigma_right = var_quantiles['2sigma']['right']
                two_sigma_range = [two_sigma_left, two_sigma_right]
                #
                three_sigma_left = var_quantiles['3sigma']['left']
                three_sigma_right = var_quantiles['3sigma']['right']
                three_sigma_range = [three_sigma_left, three_sigma_right]
                #
                five_sigma_left = var_quantiles['5sigma']['left']
                five_sigma_right = var_quantiles['5sigma']['right']
                five_sigma_range = [five_sigma_left, five_sigma_right]

                ### check for anomalous behaviour
                suspicious_dict = {}
                has_None = None in [mean, median, min_value, max_value, one_sigma_left, two_sigma_left, three_sigma_left, five_sigma_left, one_sigma_right, two_sigma_right, three_sigma_right, five_sigma_right]
                if not has_None:
                    clamp_range_left = mean + var_scaling['lim_min']*var_scaling['std']
                    clamp_range_right = mean + var_scaling['lim_max']*var_scaling['std']
                    clamp_range = [clamp_range_left, clamp_range_right]
                    suspicious_dict['left_within'] = clamp_range_left > two_sigma_left
                    suspicious_dict['right_within'] = clamp_range_right < two_sigma_right
                    # suspicious_dict['left_beyond'] = clamp_range_left < five_sigma_left
                    # suspicious_dict['right_beyond'] = clamp_range_right > five_sigma_right
                    suspicious_dict['one_sigma_empty'] = one_sigma_left == one_sigma_right
                    suspicious_dict['two_sigma_empty'] = two_sigma_left == two_sigma_right
                    suspicious_dict['three_sigma_empty'] = three_sigma_left == three_sigma_right
                    suspicious_dict['five_sigma_empty'] = five_sigma_left == five_sigma_right
                    is_suspicious = any(suspicious_dict.values())
                elif mean is None:
                    print(f'       {var_name}, {cone_type}: mean is empty')
                    continue
                else:
                    print(f'       {var_name}, {cone_type}: quantiles are empty')
                    continue

                if is_suspicious or not only_suspicious:
                    plot_ranges(file_id, var_name, cone_type, None if var_scaling_type=='linear' else mean, median, min_value, max_value,
                                clamp_range, one_sigma_range, two_sigma_range, three_sigma_range,
                                suspicious_dict, savepath=f'{output_folder}/{var_type}', close_plot=True)
                if is_suspicious:
                    print(f'-----> {var_name}, {cone_type}: looks suspicious')
                    for k, v in suspicious_dict.items():
                        print(f'           {k}: {v}')
                    print()
                else:
                    print(f'       {var_name}, {cone_type}: OK')

if __name__ == '__main__':
    main()
