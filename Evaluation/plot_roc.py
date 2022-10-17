#!/usr/bin/env python

import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mlflow
import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import OmegaConf, DictConfig

from utils.evaluation import select_curve, PlotSetup, RocCurve

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

    ref_roc = RocCurve()
    ref_roc.fill(ref_curve, create_ratio=False, ref_roc=None)

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
                    
                    roc = RocCurve()
                    if (discr_run_id==ref_discr_run_id and curve_type==ref_curve_type):
                        roc.fill(discr_curve, create_ratio=True, ref_roc=None)
                    else:
                        roc.fill(discr_curve, create_ratio='wp' not in curve_type, ref_roc=ref_roc)
                    curves_to_plot.append(roc)
                    curve_names.append(discr_cfg['name'])

        if cfg['plot_ratio']:
            fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        else:
            (fig, ax), ax_ratio = plt.subplots(1, 1, figsize=(7, 7)), None
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
        mlflow.log_artifact(path_to_pdf, cfg['output_dir'])
    print(f'\n    Saved the plot in artifacts/{cfg["output_dir"]} for runID={list(cfg.discriminators.keys())[0]}\n')

if __name__ == '__main__':
    main()
