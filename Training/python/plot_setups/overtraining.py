import distutils.util

Discriminator = None
DiscriminatorWP = None
PlotSetup = None
mode = 'public'
apply_dm_cuts = True

def Initialize(eval_tools, args):
    global Discriminator
    global DiscriminatorWP
    global PlotSetup
    global mode
    global apply_dm_cuts

    Discriminator = eval_tools.Discriminator
    DiscriminatorWP = eval_tools.DiscriminatorWP
    PlotSetup = eval_tools.PlotSetup
    if 'apply_dm_cuts' in args:
        apply_dm_cuts = distutils.util.strtobool(args['apply_dm_cuts'])

def GetDiscriminators(other_type, deep_results_label, prev_deep_results_label):
    if not deep_results_label:
        raise ValueError("Configuration parameter '--deep-results-label' must be given !!")
    deep_results_text = deep_results_label

    if not prev_deep_results_label:
        raise ValueError("Configuration parameter '--prev-deep-results-label' must be given !!")
    prev_deep_results_text = prev_deep_results_label

    if other_type == 'jet':
        discr = [            
            Discriminator(deep_results_text, 'deepId_vs_jet', True, False, 'blue'),
            Discriminator(prev_deep_results_text, 'deepId{}_vs_jet'.format(prev_deep_results_label), True, False, 'black')
        ]
        return discr
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))

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
                         ratio_ylim=[ [0.81, 1.19], [0.81, 1.19], [0.81, 1.19], [0.81, 1.19], [0.81, 1.19],
                                      [0.81, 1.19], [0.81, 1.19], [0.81, 1.19], [0.81, 1.19], [0.81, 1.19] ] )
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))
