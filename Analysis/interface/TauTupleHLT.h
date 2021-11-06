/*! Definition of a tuple with all event information that is required for the tau analysis.
*/

#pragma once

#include "TauMLTools/Core/interface/SmartTree.h"
#include <Math/VectorUtil.h>

#define VAR2(type, name1, name2) VAR(type, name1) VAR(type, name2)
#define VAR3(type, name1, name2, name3) VAR2(type, name1, name2) VAR(type, name3)
#define VAR4(type, name1, name2, name3, name4) VAR3(type, name1, name2, name3) VAR(type, name4)

#define TAU_VAR(type, name) VAR(type, tau_##name)
#define TAU_VAR2(type, name1, name2) TAU_VAR(type, name1) TAU_VAR(type, name2)
#define TAU_VAR3(type, name1, name2, name3) TAU_VAR2(type, name1, name2) TAU_VAR(type, name3)
#define TAU_VAR4(type, name1, name2, name3, name4) TAU_VAR3(type, name1, name2, name3) TAU_VAR(type, name4)

#define JET_VAR(type, name) VAR(type, jet_##name)
#define JET_VAR2(type, name1, name2) JET_VAR(type, name1) JET_VAR(type, name2)
#define JET_VAR3(type, name1, name2, name3) JET_VAR2(type, name1, name2) JET_VAR(type, name3)
#define JET_VAR4(type, name1, name2, name3, name4) JET_VAR3(type, name1, name2, name3) JET_VAR(type, name4)

#define CAND_VAR(type, name) VAR(std::vector<type>, pfCand_##name) VAR(std::vector<type>, lostTrack_##name)
#define CAND_VAR2(type, name1, name2) CAND_VAR(type, name1) CAND_VAR(type, name2)
#define CAND_VAR3(type, name1, name2, name3) CAND_VAR2(type, name1, name2) CAND_VAR(type, name3)
#define CAND_VAR4(type, name1, name2, name3, name4) CAND_VAR3(type, name1, name2, name3) CAND_VAR(type, name4)

#define TAU_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run number */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Int_t, npv) /* number of primary vertices */ \
    VAR(Float_t, rho) /* fixed grid energy density */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Int_t, sampleType) /* type of the sample (MC, Embedded or Data) */ \
    VAR(Int_t, tauType) /* tau type match e = 0, mu = 1, tau = 2, jet = 3,
                emb_e = 4, emb_mu = 5, emb_tau = 6, emb_jet = 7, data = 8 */ \
    VAR(ULong64_t, dataset_id) /* ID of the dataset (needed to identify the original dataset after shuffle&merge) */ \
    VAR(ULong64_t, dataset_group_id) /* ID of the dataset group (needed to identify the original dataset group
                                    after shuffle&merge) */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    VAR4(Float_t, pv_x, pv_y, pv_z, pv_t) /* position and time of the primary vertex (PV) */ \
    VAR4(Float_t, pv_xE, pv_yE, pv_zE, pv_tE) /* position and time errors of the primary vertex (PV) */ \
    VAR(Float_t, pv_chi2) /* chi^2 of the primary vertex (PV) */ \
    VAR(Float_t, pv_ndof) /* number of degrees of freedom of the primary vertex (PV) */ \
    VAR(Int_t, entry_index) /* Index of the entry in the event */ \
    VAR(Int_t, total_entries) /* The total number of entries in the event */ \
    /* Gen lepton with the full decay chain */ \
    VAR(Int_t, genLepton_index) /* index of the gen lepton */ \
    VAR(Int_t, genLepton_kind) /* kind of the gen lepton:
                                  Electron = 1, Muon = 2, TauElectron = 3, TauMuon = 4, Tau = 5, Other = 6 */\
    VAR(Int_t, genLepton_charge) /* charge of the gen lepton */ \
    VAR4(Float_t, genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, genLepton_vis_mass) /* visible 4-momentum of
                                                                                                 the gen lepton */ \
    VAR(Int_t, genLepton_lastMotherIndex) /* index of the last mother in genParticle_* vectors:
                                             >= 0 if at least one mother is available, -1 otherwise */ \
    VAR(std::vector<Int_t>, genParticle_pdgId) /* PDG ID */ \
    VAR(std::vector<Long64_t>, genParticle_mother) /* index of the mother. If the paricle has more than one mother,
                                                      genParticle_mother = sum(10^(3*(mother_number-1)) * mother_index).
                                                      The maximal number of mothers = 6. */ \
    VAR(std::vector<Int_t>, genParticle_charge) /* charge */ \
    VAR2(std::vector<Int_t>, genParticle_isFirstCopy, genParticle_isLastCopy) /* indicator whatever a gen particle
                                                                                 is the first or the last copy */ \
    VAR4(std::vector<Float_t>, genParticle_pt, genParticle_eta, \
                               genParticle_phi, genParticle_mass) /* 4-momenta */ \
    VAR3(std::vector<Float_t>, genParticle_vtx_x, genParticle_vtx_y, genParticle_vtx_z) /* position of the vertex */ \
    /* Gen jet variables */ \
    VAR(Int_t, genJet_index) /* index of the gen jet */ \
    VAR4(Float_t, genJet_pt, genJet_eta, genJet_phi, genJet_mass) /* 4-momentum of the gen jet */ \
    VAR(Float_t, genJet_emEnergy) /* energy of electromagnetic particles */ \
    VAR(Float_t, genJet_hadEnergy) /* energy of hadronic particles */ \
    VAR(Float_t, genJet_invisibleEnergy) /* invisible energy */ \
    VAR(Float_t, genJet_auxiliaryEnergy) /*  other energy (undecayed Sigmas etc.) */ \
    VAR(Float_t, genJet_chargedHadronEnergy) /* energy of charged hadrons */ \
    VAR(Float_t, genJet_neutralHadronEnergy) /* energy of neutral hadrons */ \
    VAR(Float_t, genJet_chargedEmEnergy) /* energy of charged electromagnetic particles */ \
    VAR(Float_t, genJet_neutralEmEnergy) /* energy of neutral electromagnetic particles */ \
    VAR(Float_t, genJet_muonEnergy) /* energy of muons */ \
    VAR(Int_t, genJet_chargedHadronMultiplicity) /* number of charged hadrons */ \
    VAR(Int_t, genJet_neutralHadronMultiplicity) /* number of neutral hadrons */ \
    VAR(Int_t, genJet_chargedEmMultiplicity) /* number of charged electromagnetic particles */ \
    VAR(Int_t, genJet_neutralEmMultiplicity) /* number of neutral electromagnetic particles */ \
    VAR(Int_t, genJet_muonMultiplicity) /* number of muons */ \
    VAR(Int_t, genJet_n_bHadrons) /* number of b hadrons clustered inside the jet */ \
    VAR(Int_t, genJet_n_cHadrons) /* number of c hadrons clustered inside the jet */ \
    VAR(Int_t, genJet_n_partons) /* number of partons clustered inside the jet */ \
    VAR(Int_t, genJet_n_leptons) /* number of leptons clustered inside the jet */ \
    VAR(Int_t, genJet_hadronFlavour) /* hadron-based flavour */ \
    VAR(Int_t, genJet_partonFlavour) /* parton-based flavour */ \
    /* Jet variables (for both AK4 and AK8 (aka "fat") jets) */ \
    JET_VAR(Int_t, index) /* index of the jet */ \
    JET_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the jet */ \
    JET_VAR(Int_t, nConstituents) /* number of jet constituents */ \
    JET_VAR2(Int_t, chargedMultiplicity, neutralMultiplicity) /* jet charged and neutral multiplicities */ \
    /* Basic tau variables */ \
    TAU_VAR(Int_t, index) /* index of the tau */ \
    TAU_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the tau */ \
    TAU_VAR(Int_t, charge) /* tau charge */ \
    /* Tau ID variables */ \
    TAU_VAR(Int_t, decayMode) /* tau decay mode */ \
    /* Tau transverse impact paramters.
       See cmssw/RecoTauTag/RecoTau/plugins/PFTauTransverseImpactParameters.cc for details */ \
    TAU_VAR3(Float_t, dxy_pca_x, dxy_pca_y, dxy_pca_z) /* The point of closest approach (PCA) of
                                                          the leadPFChargedHadrCand to the primary vertex */ \
    TAU_VAR(Float_t, dxy) /* tau signed transverse impact parameter wrt to the primary vertex */ \
    TAU_VAR(Float_t, dxy_error) /* uncertainty of the transverse impact parameter measurement */ \
    TAU_VAR(Float_t, ip3d) /* tau signed 3D impact parameter wrt to the primary vertex */ \
    TAU_VAR(Float_t, ip3d_error) /* uncertainty of the 3D impact parameter measurement */ \
    TAU_VAR(Float_t, dz) /* tau dz of the leadChargedHadrCand wrt to the primary vertex */ \
    TAU_VAR(Float_t, dz_error) /* uncertainty of the tau dz measurement */ \
    TAU_VAR(Int_t, hasSecondaryVertex) /* tau has the secondary vertex */ \
    TAU_VAR3(Float_t, sv_x, sv_y, sv_z) /* position of the secondary vertex */ \
    TAU_VAR3(Float_t, flightLength_x, flightLength_y, flightLength_z) /* flight length of the tau */ \
    TAU_VAR(Float_t, flightLength_sig) /* significance of the flight length measurement */ \
    /* Extended tau variables */ \
    TAU_VAR(Float_t, pt_weighted_deta_strip) /* sum of pt weighted values of deta relative to tau candidate
                                                for all pf photon candidates, which are associated to signal */ \
    TAU_VAR(Float_t, pt_weighted_dphi_strip) /* sum of pt weighted values of dphi relative to tau candidate
                                                for all pf photon candidates, which are associated to signal */ \
    TAU_VAR(Float_t, pt_weighted_dr_signal) /* sum of pt weighted values of dr relative to tau candidate
                                               for all pf photon candidates, which are associated to signal */ \
    TAU_VAR(Float_t, pt_weighted_dr_iso) /* sum of pt weighted values of dr relative to tau candidate
                                            for all pf photon candidates, which are inside an isolation cone
                                            but not associated to signal */ \
    TAU_VAR(Float_t, leadingTrackNormChi2) /* normalized chi2 of leading track */ \
    TAU_VAR(Float_t, e_ratio) /* ratio of energy in ECAL over sum of energy in ECAL and HCAL */ \
    TAU_VAR(Float_t, gj_angle_diff) /* Gottfried-Jackson angle difference
                                       (defined olny when the secondary vertex is reconstructed) */ \
    TAU_VAR(Int_t, n_photons) /* total number of pf photon candidates with pT>500 MeV,
                                 which are associated to signal */ \
    TAU_VAR(Float_t, emFraction) /* tau->emFraction_MVA */ \
    TAU_VAR(Int_t, inside_ecal_crack) /* tau is inside the ECAL crack (1.46 < |eta| < 1.558) */ \
    TAU_VAR(Float_t, leadChargedCand_etaAtEcalEntrance) /* eta at ECAL entrance of the leadChargedCand */ \
    /* L1 tau */ \
    VAR(Int_t, l1Tau_index) /* index of the L1 tau */ \
    VAR4(Float_t, l1Tau_pt, l1Tau_eta, l1Tau_phi, l1Tau_mass) /* L1 tau p4 */  \
    VAR2(Int_t, l1Tau_towerIEta, l1Tau_towerIPhi) /* */ \
    VAR(Int_t, l1Tau_rawEt) /* */ \
    VAR(Int_t, l1Tau_isoEt) /* */ \
    VAR(Int_t, l1Tau_hasEM) /* */ \
    VAR(Int_t, l1Tau_isMerged) /* */ \
    VAR(Int_t, l1Tau_hwIso) /* */ \
    VAR(Int_t, l1Tau_hwQual) /* */ \
    /* PF candidates and lost tracks */ \
    CAND_VAR(Int_t, index) /* index of the PF candidate */ \
    CAND_VAR(Int_t, tauSignal) /* PF candidate is a part of the tau signal */ \
    CAND_VAR(Int_t, tauLeadChargedHadrCand) /* PF candidate is the leadChargedHadrCand of the tau */ \
    CAND_VAR(Int_t, tauIso) /* PF candidate is a part of the tau isolation */ \
    CAND_VAR(Int_t, boostedTauSignal) /* PF candidate is a part of the boosted tau signal */ \
    CAND_VAR(Int_t, boostedTauLeadChargedHadrCand) /* PF candidate is the leadChargedHadrCand of the boosted tau */ \
    CAND_VAR(Int_t, boostedTauIso) /* PF candidate is a part of the boosted tau isolation */ \
    CAND_VAR(Int_t, jetDaughter) /* PF candidate is the jet daughter */ \
    CAND_VAR(Int_t, fatJetDaughter) /* PF candidate is the fat jet daughter */ \
    CAND_VAR(Int_t, subJetDaughter) /* index of the subjet of the fat jet to which PF candidate belongs
                                       (otherwise, -1) */ \
    CAND_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the PF candidate */ \
    CAND_VAR(Int_t, particleType) /* type of the PF candidate:
                                     0 - Undefined, 1 - charged hadron, 2 - electron, 3 - muon, 4 - photon,
                                     5 - neutral hadron, 6 - HF tower identified as a hadron,
                                     7 -  HF tower identified as an EM particle */ \
    CAND_VAR(Int_t, charge) /* electric charge */ \
    CAND_VAR(Int_t, lostInnerHits) /* enumerator specifying the number of lost inner hits:
                                      validHitInFirstPixelBarrelLayer = -1, noLostInnerHits = 0 (it could still not
                                      have a hit in the first layer, e.g. if it crosses an inactive sensor),
                                      oneLostInnerHit = 1, moreLostInnerHits = 2 */ \
    CAND_VAR2(Int_t, nHits, nPixelHits) /* number of the total valid hits and the number of valid pixel hits */ \
    CAND_VAR2(Int_t, nPixelLayers, nStripLayers) /* number of pixel (strip) layers with measurement */ \
    CAND_VAR4(Float_t, vertex_x, vertex_y, vertex_z, vertex_t) /* position & time of the vertex to which
                                                                  the candidate is associated */ \
    CAND_VAR2(Float_t, time, timeError) /* time and time error information on the PF candidate */ \
    CAND_VAR(Int_t, hasTrackDetails) /* has track details */ \
    CAND_VAR(Float_t, dxy) /* signed transverse impact parameter wrt to the primary vertex */ \
    CAND_VAR(Float_t, dxy_error) /* uncertainty of the transverse impact parameter measurement */ \
    CAND_VAR(Float_t, dz) /* dz wrt to the primary vertex */ \
    CAND_VAR(Float_t, dz_error) /* uncertainty of the dz measurement */ \
    CAND_VAR3(Float_t, track_pt, track_eta, track_phi) /* track momentum at the reference point */ \
    CAND_VAR(Float_t, track_chi2) /* chi^2 of the pseudo track made with the candidate kinematics */ \
    CAND_VAR(Float_t, track_ndof) /* number of degrees of freedom of the pseudo track
                                     made with the candidate kinematics */ \
    CAND_VAR2(Float_t, caloFraction, hcalFraction) /* fraction of ECAL and HCAL for HF and neutral hadrons
                                                      and isolated charged hadrons */ \
    CAND_VAR2(Float_t, rawCaloFraction, rawHcalFraction) /* raw ECAL and HCAL energy over candidate energy
                                                            for isolated charged hadrons */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_hlt, Tau, TauTuple, TAU_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_hlt, TauTuple, TAU_DATA)
#undef VAR
#undef VAR2
#undef VAR3
#undef VAR4
#undef TAU_DATA
#undef CAND_VAR
#undef CAND_VAR2
#undef CAND_VAR3
#undef CAND_VAR4

namespace tau_hlt {

template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }

struct TauTupleEntryId {
    UInt_t run;
    UInt_t lumi;
    ULong64_t evt;
    Int_t jet_index, tau_index;

    TauTupleEntryId() {}
    explicit TauTupleEntryId(const Tau& tau) :
        run(tau.run), lumi(tau.lumi), evt(tau.evt), jet_index(tau.jet_index), tau_index(tau.tau_index) {}

    bool operator<(const TauTupleEntryId& other) const
    {
        if(run != other.run) return run < other.run;
        if(lumi != other.lumi) return lumi < other.lumi;
        if(evt != other.evt) return evt < other.evt;
        if(jet_index != other.jet_index) return jet_index < other.jet_index;
        return tau_index < other.tau_index;
    }
};

} // namespace tau_tuple
