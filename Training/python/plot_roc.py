#!/usr/bin/env python

import os
import json
import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from eval_tools import select_curve, create_roc_ratio

class RocCurve:
    def __init__(self, data, ref_roc=None):
        fpr = np.array(data['false_positive_rate'])
        n_points = len(fpr)
        self.auc_score = data.get('auc_score')
        self.pr = np.empty((2, n_points))
        self.pr[0, :] = fpr
        self.pr[1, :] = data['true_positive_rate']

        if 'false_positive_rate_up' in data:
            self.pr_err = np.empty((2, 2, n_points))
            self.pr_err[0, 0, :] = data['false_positive_rate_up']
            self.pr_err[0, 1, :] = data['false_positive_rate_down']
            self.pr_err[1, 0, :] = data['true_positive_rate_up']
            self.pr_err[1, 1, :] = data['true_positive_rate_down']
        else:
            self.pr_err = None

        if 'threasholds' in data:
            self.thresholds = np.empty(n_points)
            self.thresholds[:] = data['threashods']
        else:
            self.thresholds = None

        self.color = data['plot_setup']['color']
        self.dots_only = data['plot_setup']['dots_only']
        self.dashed = data['plot_setup']['dashed']
        self.marker_size = data['plot_setup'].get('marker_size', 5)

        if ref_roc is None:
            ref_roc = self
        self.ratio = create_roc_ratio(self.pr[1], self.pr[0], ref_roc.pr[1], ref_roc.pr[0])

    def Draw(self, ax, ax_ratio = None):
        main_plot_adjusted = False
        if self.pr_err is not None:
            x = self.pr[1]
            y = self.pr[0]
            entry = ax.errorbar(x, y, xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color,
                        fmt='o', markersize=self.marker_size, linewidth=1)
        else:
            if self.dots_only:
                entry = ax.errorbar(self.pr[1], self.pr[0], color=self.color, fmt='o', markersize=self.marker_size)
            else:
                fmt = '--' if self.dashed else ''
                x = self.pr[1]
                y = self.pr[0]
                if x[-1] - x[-2] > 0.01:
                    x = x[:-1]
                    y = y[:-1]
                    x_max_main = x[-1]
                    main_plot_adjusted = True
                entry = ax.errorbar(x, y, color=self.color, fmt=fmt)
        if self.ratio is not None and ax_ratio is not None:
            if self.pr_err is not None:
                x = self.ratio[1]
                y = self.ratio[0]
                ax_ratio.errorbar(x, y, color=self.color, fmt='o', markersize='5', linewidth=1)
            else:
                linestyle = 'dashed' if self.dashed else None
                x = self.ratio[1]
                y = self.ratio[0]
                if main_plot_adjusted:
                    n = 0
                    while x[n] < x_max_main and n < len(x):
                        n += 1
                    x = x[:n]
                    y = y[:n]
                ax_ratio.plot(x, y, color=self.color, linewidth=1, linestyle=linestyle)
        return entry

class PlotSetup:
    def __init__(self, data):
        for lim_name in [ 'x', 'y', 'ratio_y' ]:
            if lim_name + '_min' in data and lim_name + '_max' in data:
                lim_value = (data[lim_name + '_min'], data[lim_name + '_max'])
            else:
                lim_value = None
            setattr(self, lim_name + 'lim', lim_value)
        self.ylabel = data.get('ylabel')
        self.yscale = data.get('yscale', 'log')
        self.ratio_yscale = data.get('ratio_yscale', 'linear')
        self.legend_loc = data.get('legend_loc', 'upper left')
        self.ratio_title = data.get('ratio_title', 'ratio')
        self.ratio_ylabel_pad = data.get('ratio_ylabel_pad', 20)

    def Apply(self, names, entries, ax, ax_ratio = None):
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_yscale(self.yscale)
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(entries, names, fontsize=14, loc=self.legend_loc)

        if ax_ratio is not None:
            if self.ratio_ylim is not None:
                ax_ratio.set_ylim(self.ratio_ylim)

            ax_ratio.set_yscale(self.ratio_yscale)
            ax_ratio.set_xlabel('Tau ID efficiency', fontsize=16)
            ax_ratio.set_ylabel(self.ratio_title, fontsize=14, labelpad=self.ratio_ylabel_pad)
            ax_ratio.tick_params(labelsize=10)
            ax_ratio.grid(True, which='both')

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path='.', config_name='plot_roc')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    experiment_id = experiment.experiment_id

    # set output paths
    output_folder = to_absolute_path(cfg.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    path_to_pdf = f'{output_folder}/{cfg.output_name}.pdf'

    # retrieve pt bin from input cfg 
    assert len(cfg.pt_bin)==2 and cfg.pt_bin[0] <= cfg.pt_bin[1]
    pt_min, pt_max = cfg.pt_bin[0], cfg.pt_bin[1]

    # retrieve reference curve
    if len(cfg.reference)>1:
        raise RuntimeError(f'Expect to have only one reference discriminator, got: {cfg.reference.keys()}')
    reference_cfg = OmegaConf.to_object(cfg.reference) # convert to python dict to enable popitem()
    ref_discr_name, ref_curve_type = reference_cfg.popitem()
    reference_json = to_absolute_path(f'mlruns/{experiment_id}/{ref_discr_name}/artifacts/performance.json')
    with open(reference_json, 'r') as f:
        ref_discr_data = json.load(f)
    ref_curve = select_curve(ref_discr_data['metrics'][ref_curve_type], 
                                pt_min=pt_min, pt_max=pt_max, vs_type=cfg.vs_type,
                                sample_tau=cfg.sample_tau, sample_vs_type=cfg.sample_vs_type)
    ref_roc = RocCurve(ref_curve, None)

    curves_to_plot = []
    curve_names = []
    with PdfPages(path_to_pdf) as pdf:
        for discr_name, curve_types in cfg.discriminators.items():
            # retrieve discriminator data from corresponding json 
            json_file = to_absolute_path(f'mlruns/{experiment_id}/{discr_name}/artifacts/performance.json')
            with open(json_file, 'r') as f:
                discr_data = json.load(f)

            for curve_type in curve_types: 
                discr_curve = select_curve(discr_data['metrics'][curve_type], 
                                            pt_min=pt_min, pt_max=pt_max, vs_type=cfg.vs_type,
                                            sample_tau=cfg.sample_tau, sample_vs_type=cfg.sample_vs_type)
                if (discr_name==ref_discr_name and curve_type==ref_curve_type) or ('wp' not in curve_type):
                    curves_to_plot.append(RocCurve(discr_curve, ref_roc=None))
                else:
                    curves_to_plot.append(RocCurve(discr_curve, ref_roc=ref_roc))
                    
                curve_names.append(discr_data['name'])

        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        plot_entries = []
        for curve_to_plot in curves_to_plot:
            plot_entry = curve_to_plot.Draw(ax, ax_ratio)
            plot_entries.append(plot_entry)

        plot_setup = PlotSetup(ref_curve['plot_setup'])
        plot_setup.Apply(curve_names, plot_entries, ax, ax_ratio)

        header_y = 1.02
        ax.text(0.03, 0.92 - len(plot_entries) * 0.10, ref_curve['plot_setup']['pt_text'], fontsize=14, transform=ax.transAxes)
        ax.text(0.01, header_y, 'CMS', fontsize=14, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
        ax.text(0.12, header_y, 'Simulation Preliminary', fontsize=14, transform=ax.transAxes, fontstyle='italic',
                fontfamily='sans-serif')
        ax.text(0.73, header_y, ref_discr_data['period'], fontsize=13, transform=ax.transAxes, fontweight='bold',
                fontfamily='sans-serif')
        plt.subplots_adjust(hspace=0)
        pdf.savefig(fig, bbox_inches='tight')

    with mlflow.start_run(experiment_id=experiment_id, run_id=list(cfg.discriminators.keys())[0]):
        mlflow.log_artifact(path_to_pdf, 'plots')

if __name__ == '__main__':
    main()