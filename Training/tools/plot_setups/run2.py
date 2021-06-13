import distutils.util

Discriminator = None
DiscriminatorWP = None
PlotSetup = None
mode = 'public'
apply_legacy_cuts = False
apply_deep_cuts = False
apply_dm_cuts = True

setup_branches = [ 'againstElectronMVA6', 'againstMuon3', 'byIsolationMVArun2017v2DBoldDMwLT2017' ]

deep_wp_thrs = {
    "e": {
        "VVVLoose": 0.0630386,
        "VVLoose": 0.1686942,
        "VLoose": 0.3628130,
        "Loose": 0.6815435,
        "Medium": 0.8847544,
        "Tight": 0.9675541,
        "VTight": 0.9859251,
        "VVTight": 0.9928449,
    },
    "mu": {
        "VLoose": 0.1058354,
        "Loose": 0.2158633,
        "Medium": 0.5551894,
        "Tight": 0.8754835,
    },
    "jet": {
        "VVVLoose": 0.2599605,
        "VVLoose": 0.4249705,
        "VLoose": 0.5983682,
        "Loose": 0.7848675,
        "Medium": 0.8834768,
        "Tight": 0.9308689,
        "VTight": 0.9573137,
        "VVTight": 0.9733927,
    },
}

def Initialize(eval_tools, args):
    global Discriminator
    global DiscriminatorWP
    global PlotSetup
    global mode
    global apply_legacy_cuts
    global apply_deep_cuts
    global apply_dm_cuts

    Discriminator = eval_tools.Discriminator
    DiscriminatorWP = eval_tools.DiscriminatorWP
    PlotSetup = eval_tools.PlotSetup
    if 'mode' in args:
        mode = args['mode']
    if 'apply_legacy_cuts' in args:
        apply_legacy_cuts = distutils.util.strtobool(args['apply_legacy_cuts'])
    if 'apply_deep_cuts' in args:
        apply_deep_cuts = distutils.util.strtobool(args['apply_deep_cuts'])
    if 'apply_dm_cuts' in args:
        apply_dm_cuts = distutils.util.strtobool(args['apply_dm_cuts'])

def GetDiscriminators(other_type, deep_results_label, prev_deep_results_label):
    deep_results_text = 'DeepTau'
    if deep_results_label is not None and len(deep_results_label) > 0:
        deep_results_text += ' ' + deep_results_label

    has_prev_results = len(prev_deep_results_label) > 0 and 'None' not in prev_deep_results_label
    if has_prev_results:
        prev_deep_results_text = deep_results_label + ' ' + prev_deep_results_label

    if other_type == 'e':
        discr = [
            Discriminator('MVA vs. electrons (JINST 13 (2018) P10005)', 'againstElectronMVA6', False, True, 'green',
                          [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                            DiscriminatorWP.VTight ] )
        ]
        if mode == 'internal':
            discr.extend([
                Discriminator('MVA6 2018', 'againstElectronMVA62018', False, True, 'red',
                              [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium,
                                DiscriminatorWP.Tight, DiscriminatorWP.VTight ] ),
                Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSeraw', True, True, 'blue',
                              [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                                DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                                DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
                              'byDeepTau2017v1VSe')
            ])
        if has_prev_results:
            discr.append(Discriminator(prev_deep_results_text,
                                       'deepId{}_vs_e'.format(prev_deep_results_label), True, False, 'black'))
        discr.append(Discriminator(deep_results_text + ' vs. electrons', 'deepId_vs_e', True, False, 'blue',
            [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose,
              DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
            working_points_thrs = deep_wp_thrs['e']))
        return discr
    elif other_type == 'mu':
        discr = [
            Discriminator('Cut based (JINST 13 (2018) P10005)', 'againstMuon3', False, True, 'green',
                          [ DiscriminatorWP.Loose, DiscriminatorWP.Tight] ),
        ]
        if mode == 'internal':
            discr.extend([
                Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSmuraw', True, True, 'blue',
                              [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                                DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                                DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
                              'byDeepTau2017v1VSmu'),
            ])
        if has_prev_results:
            discr.append(Discriminator(prev_deep_results_text, 'deepId{}_vs_mu'.format(prev_deep_results_label),
                                       True, False, 'black'))
        discr.append(Discriminator(deep_results_text + ' vs. muons', 'deepId_vs_mu', True, False, 'blue',
            [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight ],
            working_points_thrs = deep_wp_thrs['mu']))
        return discr
    elif other_type == 'jet':
        discr = [
            #'byIsolationMVArun2017v2DBoldDMwLT2017raw'
            Discriminator('MVA vs. jets (JINST 13 (2018) P10005)', 'byIsolationMVArun2017v2DBoldDMwLT2017', False, True,
                          'green',
                          [ DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose,
                            DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                            DiscriminatorWP.VVTight ],
                          'byIsolationMVArun2017v2DBoldDMwLT2017'),
            #byIsolationMVArun2017v2DBnewDMwLT2017raw
            Discriminator('MVA (updated decay modes)', 'byIsolationMVArun2017v2DBnewDMwLT2017', False, True, 'red',
                          [ DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose,
                            DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                            DiscriminatorWP.VVTight ],
                          'byIsolationMVArun2017v2DBnewDMwLT2017', dashed=True),

        ]
        if mode == 'internal':
            discr.extend([
                Discriminator('DPF 2016v0', 'byDpfTau2016v0VSallraw', True, True, 'magenta', [ DiscriminatorWP.Tight ],
                              'byDpfTau2016v0VSall'),
                Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSjetraw', True, True, 'blue',
                              [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                                DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                                DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
                              'byDeepTau2017v1VSjet')
            ])
        if has_prev_results:
            discr.append(Discriminator(prev_deep_results_text, 'deepId{}_vs_jet'.format(prev_deep_results_label),
                                       True, False, 'black'))
        discr.append(Discriminator(deep_results_text + ' vs. jets', 'deepId_vs_jet', True, False, 'blue',
            [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose,
              DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
            working_points_thrs = deep_wp_thrs['jet']))
        return discr
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))

def ApplySelection(df, other_type):
    if apply_legacy_cuts:
        if other_type == 'e':
            df = df[ \
                (np.bitwise_and(df['byIsolationMVArun2017v2DBoldDMwLT2017'], 1 << DiscriminatorWP.VVLoose) > 0) \
                & (np.bitwise_and(df['againstMuon3'], 1 << DiscriminatorWP.Loose) > 0) \
                & (df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6) ]
        elif other_type == 'mu':
            df = df[ \
                (np.bitwise_and(df['byIsolationMVArun2017v2DBoldDMwLT2017'], 1 << DiscriminatorWP.VVLoose) > 0) \
                & (np.bitwise_and(df['againstElectronMVA6'], 1 << DiscriminatorWP.VLoose) > 0) \
                & (df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6) ]
        elif other_type == 'jet':
            df = df[ (np.bitwise_and(df['againstElectronMVA6'], 1 << DiscriminatorWP.VLoose) > 0) \
                             & (np.bitwise_and(df['againstMuon3'], 1 << DiscriminatorWP.Loose) > 0) \
                             & (df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6) ]
    elif apply_deep_cuts:
        if other_type == 'e':
            df = df[(df['deepId_vs_jet'] > deep_wp_thrs['jet']['VVVLoose']) \
                & (df['deepId_vs_mu'] > deep_wp_thrs['mu']['VLoose']) ]
        elif other_type == 'mu':
            df = df[(df['deepId_vs_e'] > deep_wp_thrs['e']['VVVLoose']) \
                & (df['deepId_vs_jet'] > deep_wp_thrs['jet']['VVVLoose'])]
        elif other_type == 'jet':
            df = df[ (df['deepId_vs_e'] > deep_wp_thrs['e']['VVVLoose']) \
                & (df['deepId_vs_mu'] > deep_wp_thrs['mu']['VLoose'])]
    elif apply_dm_cuts:
        df = df[(df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6)]
        #df = df[(df['tau_decayModeFinding'] < 0.5) & (df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6)]
        #df = df[(df['tau_decayModeFinding'] > 0.5)]
        #df = df[((df['tau_decayMode'] != 5) & (df['tau_decayMode'] != 6)) \
        #                | (df['gen_tau'] == 0) \
        #                | (df['tau_charge'].values.astype(int) == df['lepton_gen_charge'])]
    return df

def GetPtBins():
    if mode == 'internal':
        return [ 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000 ]
    return [ 20, 100, 1000 ]

def GetPlotSetup(other_type):
    if other_type == 'e':
        return PlotSetup(ylabel='Electron mis-id probability', xlim=[0.5, 1], ratio_yscale='log',
                         ylim=[2e-5, 1], ratio_ylim=[0.5, 40])
    elif other_type == 'mu':
        return PlotSetup(ylabel='Muon mis-id probability', xlim=[0.98, 1], ratio_yscale='log',
                         ylim=[2e-5, 1],
                         ratio_ylim=[ [0.5, 20], [0.5, 20], [0.5, 20], [0.5, 20], [0.5, 20],
                                      [0.5, 20], [0.5, 20], [0.5, 50], [0.5, 50], [0.5, 50] ] )
    elif other_type == 'jet':
        return PlotSetup(ylabel='Jet mis-id probability', ratio_ylabel_pad=30, xlim=[0.3, 1],
                         # ylim=[ [2e-3, 1], [3e-4, 1], [8e-5, 1], [2e-5, 1], [2e-5, 1],
                         #        [5e-6, 1], [5e-6, 1], [5e-6, 1], [5e-6, 1], [2e-6, 1] ],
                         ylim=[ [1e-3, 1], [2e-4, 1], [8e-5, 1], [2e-5, 1], [2e-5, 1],
                                [5e-6, 1], [5e-6, 1], [5e-6, 1], [5e-6, 1], [2e-6, 1] ],
                         ratio_ylim=[ [0.5, 4.5], [0.5, 6.5], [0.5, 2.5], [0.5, 2.5], [0.5, 2.5],
                                      [0.5, 3.5], [0.5, 3.5], [0.5, 3.5], [0.5, 10], [0.5, 10] ] )
                         # ratio_ylim=[ [0.5, 3], [0.5, 5], [0.5, 2.5], [0.5, 2.5], [0.5, 2.5],
                         #              [0.5, 3.5], [0.5, 3.5], [0.5, 3.5], [0.5, 10], [0.5, 10] ] )
    else:
        raise RuntimeError('Unknown other_type = "{}"'.format(other_type))
