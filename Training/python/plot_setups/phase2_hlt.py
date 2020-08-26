import distutils.util
import pandas
import math

Discriminator = None
DiscriminatorWP = None
PlotSetup = None
apply_dm_cuts = True

setup_branches = [ 'spectator_chargedIsoPtSumHGCalFix' ]

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
            Discriminator('charged iso', 'score_chargedIsoPtSumHGCalFix', True, False, 'green',
                          [ DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight ],
                          working_points_thrs = { "Loose": -0.20, "Medium": -0.10, "Tight": -0.05 }),
        ]
        if has_prev_results:
            discr.append(Discriminator(prev_deep_results_text, 'deepId{}_vs_jet'.format(prev_deep_results_label),
                                       True, False, 'black'))
        ##discr.append(Discriminator(deep_results_text + ' vs. jets', 'deepId_vs_jet', True, False, 'blue'))
        discr.append(Discriminator(deep_results_text + ' vs. jets', 'deepId_vs_jet', True, False, 'blue',
                                   [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose,
                                     DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
                                   working_points_thrs = { "VVVLoose": 0.2599605, "VVLoose": 0.4249705, "VLoose": 0.5983682, "Loose": 0.7848675,
                                                           "Medium": 0.8834768, "Tight": 0.9308689, "VTight": 0.9573137, "VVTight": 0.9733927 }))
        return discr
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))

def DefineBranches(df, tau_types):
    values = []
    for i in range(len(df.index)):
        values.append(-df.spectator_chargedIsoPtSumHGCalFix[i]/max(1.e-9, df.tau_pt[i]))
    df['score_chargedIsoPtSumHGCalFix'] = pandas.Series(values, index=df.index)
    return df

def ApplySelection(df, other_type):
    if apply_dm_cuts:
        df = df[(df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6)]
    return df

def GetPtBins():
    return [ 20, 30, 40, 60, 80, 120, 1000 ]

def GetPlotSetup(other_type):
    if other_type == 'jet':
        return PlotSetup(ylabel='Jet mis-id probability', ratio_ylable_pad=30, xlim=[0, 1],
                         ylim=[ [1e-3, 1], [2e-4, 1], [8e-5, 1], [2e-5, 1], [2e-5, 1],
                                [5e-6, 1], [5e-6, 1], [5e-6, 1], [5e-6, 1], [2e-6, 1] ],
                         ratio_ylim=[ [0.5, 7.5], [0.5, 7.5], [0.5, 7.5], [0.5, 7.5], [0.5, 7.5],
                                      [0.5, 7.5], [0.5, 7.5] ] )
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))
