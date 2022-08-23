import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib
import matplotlib.pyplot as plt

def efficiency(probs, N, thrs):
    eff, eff_down, eff_up = np.zeros(thrs.shape[0]), np.zeros(thrs.shape[0]), np.zeros(thrs.shape[0])
    if N != 0:
        for k in range(thrs.shape[0]):
            n = np.count_nonzero(probs > thrs[k])
            eff[k] = n/N
            eff_down[k], eff_up[k] = proportion_confint(n, N, 1-0.682689, 'beta')
    return eff, eff_down, eff_up

def differential_efficiency(df_true, df_fake, var_name, var_bins, 
                            vs_type, discr_column_prefix, thrs, 
                            require_WPs_in_numerator, require_WPs_in_denominator, WPs_to_require, wp_definitions):
    var_bins, thrs = np.array(var_bins), np.array(thrs)
    eff = np.zeros([2, len(var_bins)-1, thrs.shape[0]])
    eff_up, eff_down = np.zeros(eff.shape), np.zeros(eff.shape)

    # compose a string with WP cuts
    if require_WPs_in_numerator or require_WPs_in_denominator:
        WP_cut = []
        for other_vs_type, wp_to_require in WPs_to_require.items():
            thr = wp_definitions[other_vs_type][wp_to_require]
            cut = f'({discr_column_prefix}{other_vs_type} > {thr})'
            WP_cut.append(cut)
            if require_WPs_in_denominator:
                print(f'\n-> After passing required {wp_to_require} WP vs. {other_vs_type}:')
                for tau_type, df in zip(['tau', 'fakes'], [df_true, df_fake]):
                    print('    ', tau_type, ': ', df.query(cut, inplace=False).shape[0])
        WP_cut = ' and '.join(WP_cut)
        if require_WPs_in_denominator:
            print(f'\n-> After passing all required WPs:')
            for tau_type, df in zip(['tau', 'fakes'], [df_true, df_fake]):
                print('    ', tau_type, ': ', df.query(WP_cut, inplace=False).shape[0])
    else:
        WP_cut = None

    # select objects passing WP requirement as a base (denominator) of eff computation 
    if require_WPs_in_denominator:
        df_true = df_true.query(WP_cut, inplace=False)
        df_fake = df_fake.query(WP_cut, inplace=False)
        assert require_WPs_in_numerator, "require_WPs_in_numerator should be True, if require_WPs_in_denominator==True"

    for i, (var_min, var_max) in enumerate(zip(var_bins[:-1], var_bins[1:])):
        # slice dataframes in given bin
        df_bin_true = df_true.query(f'{var_name}>{var_min} and {var_name}<={var_max}', inplace=False)
        df_bin_fake = df_fake.query(f'{var_name}>{var_min} and {var_name}<={var_max}', inplace=False)
        N_true = df_bin_true.shape[0]
        N_fake = df_bin_fake.shape[0]
        if require_WPs_in_numerator: # additionally require to pass specified WPs for eff's numerator
            df_bin_true = df_bin_true.query(WP_cut, inplace=False)
            df_bin_fake = df_bin_fake.query(WP_cut, inplace=False)
        eff[0, i, :], eff_down[0, i, :], eff_up[0, i, :] = efficiency(df_bin_true[f'{discr_column_prefix}{vs_type}'], N_true, thrs)
        eff[1, i, :], eff_down[1, i, :], eff_up[1, i, :] = efficiency(df_bin_fake[f'{discr_column_prefix}{vs_type}'], N_fake, thrs)
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