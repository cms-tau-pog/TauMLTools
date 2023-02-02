from numba.typed import Dict
from numba import njit
import vector

@njit
def get_genLepton_match(genLepton_match_map, genLepton_kind_map, genLepton_index, genJet_index, genLepton_kind, genLepton_vis_pt, is_dR_matched):
    
    if genLepton_index >= 0:
        if not is_dR_matched:
            return genLepton_match_map['None'] 
        else:
            if (genLepton_kind == genLepton_kind_map['PromptElectron']):
                if (genLepton_vis_pt < 8.0):
                    return genLepton_match_map['None'] 
                else:
                    return genLepton_match_map['Electron']
                
            elif (genLepton_kind == genLepton_kind_map['PromptMuon']):
                if (genLepton_vis_pt < 8.0):
                    return genLepton_match_map['None'] 
                else:
                    return genLepton_match_map['Muon']
            
            elif (genLepton_kind == genLepton_kind_map['TauDecayedToElectron']):
                if (genLepton_vis_pt < 8.0):
                    return genLepton_match_map['None'] 
                else:
                    return genLepton_match_map['TauElectron']
            
            elif (genLepton_kind == genLepton_kind_map['TauDecayedToMuon']):
                if (genLepton_vis_pt < 8.0):
                    return genLepton_match_map['None'] 
                else:
                    return genLepton_match_map['TauMuon']
            
            elif (genLepton_kind == genLepton_kind_map['TauDecayedToHadrons']):
                if (genLepton_vis_pt < 15.0):
                    return genLepton_match_map['None'] 
                else:
                    return genLepton_match_map['Tau']
            else: 
                return genLepton_match_map['Exception']
                
    elif genJet_index >= 0:
        return genLepton_match_map['NoMatch'] 
    else:
        return genLepton_match_map['None'] 

@njit
def recompute_tau_type(genLepton_match_map, genLepton_kind_map, sample_type_map, tau_type_map,
                       sample_type, is_dR_matched,
                       genLepton_index, genJet_index, genLepton_kind, genLepton_vis_pt):
    tau_types = []

    for i in range(len(genLepton_index)): # loop over taus
        gen_match = get_genLepton_match(genLepton_match_map, genLepton_kind_map, 
                                genLepton_index[i], genJet_index[i], genLepton_kind[i], genLepton_vis_pt[i], is_dR_matched[i])

        if sample_type[i]==sample_type_map['MC'] and (gen_match==genLepton_match_map['Electron'] or gen_match==genLepton_match_map['TauElectron']):
            tau_types.append(tau_type_map['e'])
            
        elif sample_type[i]==sample_type_map['MC'] and (gen_match==genLepton_match_map['Muon'] or gen_match==genLepton_match_map['TauMuon']):
            tau_types.append(tau_type_map['mu'])
            
        elif gen_match==genLepton_match_map['Tau']:
            if sample_type[i]==sample_type_map['MC']: tau_types.append(tau_type_map['tau'])
            if sample_type[i]==sample_type_map['Embedded']:  tau_types.append(tau_type_map['emb_tau'])

        elif gen_match==genLepton_match_map['NoMatch']:
            if(sample_type[i]==sample_type_map['MC']): tau_types.append(tau_type_map['jet'])
            if(sample_type[i]==sample_type_map['Data']): tau_types.append(tau_type_map['data'])
            if(sample_type[i]==sample_type_map['Embedded']): tau_types.append(tau_type_map['emb_jet'])

        elif sample_type[i]==sample_type_map['Embedded']:
            if(gen_match==genLepton_match_map['TauMuon']): tau_types.append(tau_type_map['emb_mu'])
            if(gen_match==genLepton_match_map['TauElectron']): tau_types.append(tau_type_map['emb_e'])
        else:
            tau_types.append(tau_type_map['no_type'])

    return tau_types

def compute_genmatch_dR(a):
    tau_v = vector.Array({'pt': a['tau_pt'], 'eta': a['tau_eta'], 'phi': a['tau_phi']})
    genlepton_v = vector.Array({'pt': a['genLepton_vis_pt'], 'eta': a['genLepton_vis_eta'], 'phi': a['genLepton_vis_phi']})
    dR = tau_v.deltaR(genlepton_v)
    return dR

def dict_to_numba(d, key_type, value_type):
    d_numba = Dict.empty(key_type=key_type, value_type=value_type)
    for k,v in d.items():
        d_numba[k] = v
    return d_numba
