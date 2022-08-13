#!/usr/bin/env python

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mlflow
import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import OmegaConf, DictConfig

from eval_tools import select_curve, create_roc_ratio, PlotSetup

class RocCurve:
    def __init__(self, data, ref_roc=None, WPcurve=False):
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

        if 'thresholds' in data:
            self.thresholds = np.empty(n_points)
            self.thresholds[:] = data['thresholds']
        else:
            self.thresholds = None

        self.color = data['plot_cfg']['color']
        self.alpha = data['plot_cfg'].get('alpha', 1.)
        self.dots_only = data['plot_cfg']['dots_only']
        self.dashed = data['plot_cfg']['dashed']
        self.marker_size = data['plot_cfg'].get('marker_size', 5)

        if ref_roc is None:
            ref_roc = self

        if WPcurve:
            self.ratio = None
        else:
            self.ratio = create_roc_ratio(self.pr[1], self.pr[0], ref_roc.pr[1], ref_roc.pr[0], True)

    def draw(self, ax, ax_ratio = None):
        main_plot_adjusted = False
        if self.pr_err is not None:
            x = self.pr[1]
            y = self.pr[0]
            entry = ax.errorbar(x, y, xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color, alpha=self.alpha,
                        fmt='o', markersize=self.marker_size, linewidth=1)
        else:
            if self.dots_only:
                entry = ax.errorbar(self.pr[1], self.pr[0], color=self.color, alpha=self.alpha, fmt='o', markersize=self.marker_size)
            else:
                fmt = '--' if self.dashed else ''
                x = self.pr[1]
                y = self.pr[0]
                if x[-1] - x[-2] > 0.01:
                    x = x[:-1]
                    y = y[:-1]
                    x_max_main = x[-1]
                    main_plot_adjusted = True
                entry = ax.errorbar(x, y, color=self.color, alpha=self.alpha, fmt=fmt)
        if self.ratio is not None and ax_ratio is not None:
            if self.pr_err is not None:
                x = self.ratio[1]
                y = self.ratio[0]
                ax_ratio.errorbar(x, y, color=self.color, alpha=self.alpha, fmt='o', markersize='5', linewidth=1)
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
                ax_ratio.plot(x, y, color=self.color, alpha=self.alpha, linewidth=1, linestyle=linestyle)
        return entry

@hydra.main(config_path='configs', config_name='plot_roc')
def main(cfg: DictConfig) -> None:
    path_to_mlflow = to_absolute_path(cfg.path_to_mlflow)
    mlflow.set_tracking_uri(f"file://{path_to_mlflow}")
    dmname = '_'.join([str(x) for x in cfg.dm_bin])
    path_to_pdf = f'./{cfg.output_name}{dmname}.pdf' # hydra log directory
    print()

    # retrieve pt bin from input cfg 
    assert len(cfg.pt_bin)==2 and cfg.pt_bin[0] <= cfg.pt_bin[1]
    pt_min, pt_max = cfg.pt_bin[0], cfg.pt_bin[1]
    assert len(cfg.eta_bin)==2 and cfg.eta_bin[0] <= cfg.eta_bin[1]
    eta_min, eta_max = cfg.eta_bin[0], cfg.eta_bin[1]
    assert len(cfg.dm_bin)>=1
    dm_bin = cfg.dm_bin

    # retrieve reference curve
    if len(cfg.reference)>1:
        raise RuntimeError(f'Expect to have only one reference discriminator, got: {cfg.reference.keys()}')
    reference_cfg = OmegaConf.to_object(cfg.reference) # convert to python dict to enable popitem()
    ref_discr_run_id, ref_curve_type = reference_cfg.popitem()
    ref_discr_cfg = cfg["discriminators"][ref_discr_run_id]
    assert isinstance(ref_curve_type, str)

    reference_json = f'{path_to_mlflow}/{cfg.experiment_id}/{ref_discr_run_id}/artifacts/performance.json'
    with open(reference_json, 'r') as f:
        ref_discr_data = json.load(f)
    ref_curve = select_curve(ref_discr_data['metrics'][ref_curve_type], 
                                pt_min=pt_min, pt_max=pt_max, eta_min=eta_min, eta_max=eta_max, dm_bin=dm_bin, vs_type=cfg.vs_type,
                                dataset_alias=cfg.dataset_alias)
    if ref_curve is None:
        raise RuntimeError('[INFO] didn\'t manage to retrieve a reference curve from performance.json')

    # import plotting parameters from plot_roc.yaml to class init kwargs
    ref_curve['plot_cfg'] = ref_discr_cfg['plot_cfg']

    ref_roc = RocCurve(ref_curve, None)

    curves_to_plot = []
    curve_names = []
    with PdfPages(path_to_pdf) as pdf:
        for discr_run_id, discr_cfg in cfg.discriminators.items():
            # retrieve discriminator data from corresponding json 
            json_file = f'{path_to_mlflow}/{cfg.experiment_id}/{discr_run_id}/artifacts/performance.json'
            with open(json_file, 'r') as f:
                discr_data = json.load(f)

            for curve_type in discr_cfg["curve_types"]: 
                discr_curve = select_curve(discr_data['metrics'][curve_type], 
                                            pt_min=pt_min, pt_max=pt_max, eta_min=eta_min, eta_max=eta_max, dm_bin=dm_bin, vs_type=cfg.vs_type,
                                            dataset_alias=cfg.dataset_alias)
                if discr_curve is None:
                    print(f'[INFO] Didn\'t manage to retrieve a curve ({curve_type}) for discriminator ({discr_run_id}) from performance.json. Will proceed without plotting it.')
                    continue
                else:
                    discr_curve['plot_cfg'] = discr_cfg['plot_cfg']
                    if 'wp' in curve_type:
                        discr_curve['plot_cfg']['dots_only'] = True
                    # elif (discr_run_id==ref_discr_run_id and curve_type==ref_curve_type) or ('wp' in curve_type and any('curve' in ctype for ctype in discr_cfg["curve_types"])): # Temporary: Don't make ratio for 'roc_wp' if there's a ratio for 'roc_curve' already
                    if (discr_run_id==ref_discr_run_id and curve_type==ref_curve_type):
                        roc = RocCurve(discr_curve, ref_roc=None)
                    else:
                        roc = RocCurve(discr_curve, ref_roc=ref_roc, WPcurve='wp' in curve_type)
                    curves_to_plot.append(roc)
                    curve_names.append(discr_cfg['name'])

        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        plot_entries = []
        for curve_to_plot in curves_to_plot:
            plot_entry = curve_to_plot.draw(ax, ax_ratio)
            plot_entries.append(plot_entry)

        # apply plotting style & save
        plot_setup = instantiate(cfg['plot_setup'])
        plot_setup.apply(curve_names, plot_entries, ax, ax_ratio)
        plot_setup.add_text(ax, len(set(curve_names)), pt_min, pt_max, eta_min, eta_max, dm_bin, cfg['period'])
        plt.subplots_adjust(hspace=0)
        pdf.savefig(fig, bbox_inches='tight')

    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=list(cfg.discriminators.keys())[0]):
        mlflow.log_artifact(path_to_pdf, 'plots')
    print(f'\n    Saved the plot in artifacts/plots for runID={list(cfg.discriminators.keys())[0]}\n')

if __name__ == '__main__':
    main()
