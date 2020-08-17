import distutils.util
import pandas

Discriminator = None
DiscriminatorWP = None
PlotSetup = None
apply_dm_cuts = True

setup_branches = [ 'chargedIsoPtSum' ]

def Initialize(eval_tools, args):
    global Discriminator
    global DiscriminatorWP
    global PlotSetup
    global apply_dm_cuts

    Discriminator = eval_tools.Discriminator
    DiscriminatorWP = eval_tools.DiscriminatorWP
    PlotSetup = eval_tools.PlotSetup
    if 'apply_dm_cuts' in args:
        apply_dm_cuts = distutils.util.strtobool(args['apply_dm_cuts'])

def GetDiscriminators(other_type, deep_results_label, prev_deep_results_label):
    deep_results_text = 'DeepTau'
    if deep_results_label is not None and len(deep_results_label) > 0:
        deep_results_text += ' ' + deep_results_label

    has_prev_results = len(prev_deep_results_label) > 0 and 'None' not in prev_deep_results_label
    if has_prev_results:
        prev_deep_results_text = deep_results_label + ' ' + prev_deep_results_label

    if other_type == 'jet':
        discr = [
            Discriminator('charged iso', 'relNegChargedIsoPtSum', True, False, 'green',
                          [ DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight ],
                          working_points_thrs = { "Loose": -0.2, "Medium": -0.1, "Tight": -0.05 }),
        ]
        if has_prev_results:
            discr.append(Discriminator(prev_deep_results_text, 'deepId{}_vs_jet'.format(prev_deep_results_label),
                                       True, False, 'black'))
        discr.append(Discriminator(deep_results_text + ' vs. jets', 'deepId_vs_jet', True, False, 'blue'))
        return discr
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))

def DefineBranches(df, tau_types):
    df['chargedIsoPtSum'] = pandas.Series(df.chargedIsoPtSum * 123.5 + 47.78, index=df.index)
    df['relNegChargedIsoPtSum'] = pandas.Series(-df.chargedIsoPtSum / df.tau_pt, index=df.index)
    return df

def ApplySelection(df, other_type):
    if apply_dm_cuts:
        df = df[(df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6)]
    return df

def GetPtBins():
    return [ 20, 100, 1000 ]

def GetPlotSetup(other_type):
    if other_type == 'jet':
        return PlotSetup(ylabel='Jet mis-id probability', ratio_ylable_pad=30, xlim=[0.3, 1],
                         ylim=[ [1e-3, 1], [2e-4, 1], [8e-5, 1], [2e-5, 1], [2e-5, 1],
                                [5e-6, 1], [5e-6, 1], [5e-6, 1], [5e-6, 1], [2e-6, 1] ],
                         ratio_ylim=[ [0.5, 4.5], [0.5, 6.5], [0.5, 2.5], [0.5, 2.5], [0.5, 2.5],
                                      [0.5, 3.5], [0.5, 3.5], [0.5, 3.5], [0.5, 10], [0.5, 10] ] )
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))
