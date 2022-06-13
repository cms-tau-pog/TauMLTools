import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib
import matplotlib.pyplot as plt

def efficiency(probs, thrs):
    N = probs.shape[0]
    eff, eff_down, eff_up = np.zeros(thrs.shape[0]), np.zeros(thrs.shape[0]), np.zeros(thrs.shape[0])
    if N != 0:
        for k in range(thrs.shape[0]):
            n = np.count_nonzero(probs > thrs[k])
            eff[k] = n/N
            eff_down[k], eff_up[k] = proportion_confint(n, N, 1-0.682689, 'beta')
    return eff, eff_down, eff_up

def differential_efficiency(df_true, df_fake, var_name, var_bins, discr_column_name, thrs):
    var_bins, thrs = np.array(var_bins), np.array(thrs)
    eff = np.zeros([2, len(var_bins)-1, thrs.shape[0]])
    eff_up, eff_down = np.zeros(eff.shape), np.zeros(eff.shape)
    for i, (var_min, var_max) in enumerate(zip(var_bins[:-1], var_bins[1:])):
        x_true = df_true.query(f'{var_name}>{var_min} and {var_name}<={var_max}')[discr_column_name]
        x_fake = df_fake.query(f'{var_name}>{var_min} and {var_name}<={var_max}')[discr_column_name]
        eff[0, i, :], eff_down[0, i, :], eff_up[0, i, :] = efficiency(x_true, thrs)
        eff[1, i, :], eff_down[1, i, :], eff_up[1, i, :] = efficiency(x_fake, thrs)
    eff_up = eff_up - eff
    eff_down = eff - eff_down
    return eff, eff_up, eff_down

def plot_efficiency(eff, eff_up, eff_down, labels, var_bins, 
                    xscale, yscale_1, yscale_2, ylim_1, ylim_2,
                    xtitle, ytitle_1, ytitle_2, legend_loc, 
                    xlabels=None, xerr=None, **kwargs):
    var_bins = np.array(var_bins)
    if xlabels is None:
        x = (var_bins[1:]+var_bins[0:-1])/2
    else:
        x = range(len(xlabels))
    if xerr is None:
        xerr = ((var_bins[1:]-var_bins[0:-1])/2)
    ncol = len(labels) // 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    for k in range(len(labels)):
        ax1.errorbar(x, eff[0, :, k], xerr=xerr, yerr=[eff_down[0, :, k], eff_up[0, :, k]], linestyle='None')
        ax2.errorbar(x, eff[1, :, k], xerr=xerr, yerr=[eff_down[1, :, k], eff_up[1, :, k]], linestyle='None')
            
    ax1.set_xlabel(xtitle, fontsize=16)
    ax1.set_xscale(xscale)
    ax1.set_yscale(yscale_1)
    ax1.set_ylabel(ytitle_1, fontsize=16)
    ax1.set_ylim(ylim_1)
    ax1.tick_params(labelsize=14)
    ax1.grid(True)
    ax1.legend(labels, fontsize=14, loc=legend_loc, ncol=ncol)
    if yscale_1 == 'log':
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.98,0.99,0.995),numticks=12)
        ax1.yaxis.set_minor_locator(locmin)
        #ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    ax2.set_ylim(ylim_2)
    ax2.set_xscale(xscale)
    ax2.set_yscale(yscale_2)
    ax2.set_xlabel(xtitle, fontsize=16)
    ax2.set_ylabel(ytitle_2, fontsize=16)
    ax2.tick_params(labelsize=14)
    ax2.grid(True)
    ax2.legend(labels, fontsize=14, loc=legend_loc, ncol=ncol)
    
    if xlabels is None:
        ax1.set_xlim([var_bins[0], var_bins[-1]])
        ax2.set_xlim([var_bins[0], var_bins[-1]])
    else:
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels)
        ax2.set_xticks(x)
        ax2.set_xticklabels(xlabels)

    plt.show()
    return fig