/*! Definition of a tuple with all event information that is required for the tau analysis.
*/

#pragma once

#include "TauMLTools/Core/interface/SmartTree.h"
#include "TauMLTools/Analysis/interface/TauIdResults.h"
#include <Math/VectorUtil.h>

#define VAR2(type, name1, name2) VAR(type, name1) VAR(type, name2)
#define VAR3(type, name1, name2, name3) VAR2(type, name1, name2) VAR(type, name3)
#define VAR4(type, name1, name2, name3, name4) VAR3(type, name1, name2, name3) VAR(type, name4)

#define TAU_VAR(type, name) VAR(type, tau_##name) VAR(type, boostedTau_##name)
#define TAU_VAR2(type, name1, name2) TAU_VAR(type, name1) TAU_VAR(type, name2)
#define TAU_VAR3(type, name1, name2, name3) TAU_VAR2(type, name1, name2) TAU_VAR(type, name3)
#define TAU_VAR4(type, name1, name2, name3, name4) TAU_VAR3(type, name1, name2, name3) TAU_VAR(type, name4)
#define TAU_ID(name, pattern, has_raw, wp_list) TAU_VAR(uint16_t, name) TAU_VAR(Float_t, name##raw)

#define JET_VAR(type, name) VAR(type, jet_##name) VAR(type, fatJet_##name)
#define JET_VAR2(type, name1, name2) JET_VAR(type, name1) JET_VAR(type, name2)
#define JET_VAR3(type, name1, name2, name3) JET_VAR2(type, name1, name2) JET_VAR(type, name3)
#define JET_VAR4(type, name1, name2, name3, name4) JET_VAR3(type, name1, name2, name3) JET_VAR(type, name4)

#define CAND_VAR(type, name) VAR(std::vector<type>, pfCand_##name) VAR(std::vector<type>, lostTrack_##name)
#define CAND_VAR2(type, name1, name2) CAND_VAR(type, name1) CAND_VAR(type, name2)
#define CAND_VAR3(type, name1, name2, name3) CAND_VAR2(type, name1, name2) CAND_VAR(type, name3)
#define CAND_VAR4(type, name1, name2, name3, name4) CAND_VAR3(type, name1, name2, name3) CAND_VAR(type, name4)

#define ELE_VAR(type, name) VAR(std::vector<type>, ele_##name)
#define ELE_VAR2(type, name1, name2) ELE_VAR(type, name1) ELE_VAR(type, name2)
#define ELE_VAR3(type, name1, name2, name3) ELE_VAR2(type, name1, name2) ELE_VAR(type, name3)
#define ELE_VAR4(type, name1, name2, name3, name4) ELE_VAR3(type, name1, name2, name3) ELE_VAR(type, name4)
#define ELE_VAR5(type, name1, name2, name3, name4, name5) ELE_VAR4(type, name1, name2, name3, name4) \
                                                          ELE_VAR(type, name5)

#define MUON_VAR(type, name) VAR(std::vector<type>, muon_##name)
#define MUON_VAR2(type, name1, name2) MUON_VAR(type, name1) MUON_VAR(type, name2)
#define MUON_VAR3(type, name1, name2, name3) MUON_VAR2(type, name1, name2) MUON_VAR(type, name3)
#define MUON_VAR4(type, name1, name2, name3, name4) MUON_VAR3(type, name1, name2, name3) MUON_VAR(type, name4)

#define TRACK_VAR(type, name) VAR(std::vector<type>, isoTrack_##name)
#define TRACK_VAR2(type, name1, name2) TRACK_VAR(type, name1) TRACK_VAR(type, name2)
#define TRACK_VAR3(type, name1, name2, name3) TRACK_VAR2(type, name1, name2) TRACK_VAR(type, name3)
#define TRACK_VAR4(type, name1, name2, name3, name4) TRACK_VAR3(type, name1, name2, name3) TRACK_VAR(type, name4)

#define TAU_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run number */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Int_t, npv) /* number of primary vertices */ \
    VAR(Float_t, rho) /* fixed grid energy density */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Float_t, trainingWeight) /* training weight */ \
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
    VAR2(Float_t, met_pt, met_phi) /* MET momentum */ \
    VAR3(Float_t, metcov_00, metcov_01, metcov_11) /* MET covariance */ \
    VAR2(Float_t, puppimet_pt, puppimet_phi) /* PuppiMET momentum */ \
    VAR3(Float_t, puppimetcov_00, puppimetcov_01, puppimetcov_11) /* MET covariance */ \
    VAR2(Float_t, deepmet_pt, deepmet_phi) /* DeepMET momentum */ \
    VAR3(Float_t, deepmetcov_00, deepmetcov_01, deepmetcov_11) /* MET covariance */ \
    VAR2(Float_t, genmet_pt, genmet_phi) /* DeepMET momentum */ \
    VAR(Int_t, entry_index) /* Index of the entry in the event */ \
    VAR(Int_t, total_entries) /* The total number of entries in the event */ \
    /* Gen lepton with the full decay chain */ \
    /* TauSpinner weights */ \
    VAR(Double_t, tauSpinnerWTEven) \
    VAR(Double_t, tauSpinnerWTOdd) \
    VAR(Double_t, tauSpinnerWTMM) \
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
    /* Tag object variables (for tag-and-probe data) */ \
    VAR(Int_t, tagObj_valid) /* indicates presence of the tag object */ \
    VAR4(Float_t, tagObj_pt, tagObj_eta, tagObj_phi, tagObj_mass) /* 4-momentum of the tag object */ \
    VAR(Int_t, tagObj_charge) /* charge of the tag object */ \
    VAR(UInt_t, tagObj_id) /* ID of the tag object */ \
    VAR(Float_t, tagObj_iso) /* isolation of the tag object */ \
    VAR(Int_t, has_extramuon) /* Extra muon present */ \
    VAR(Int_t, has_extraelectron) /* Extra electron present */ \
    VAR(Int_t, has_dimuon) /* Extra muon pair present */ \
    /* Jet variables (for both AK4 and AK8 (aka "fat") jets) */ \
    JET_VAR(Int_t, index) /* index of the jet */ \
    JET_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the jet */ \
    JET_VAR(Float_t, neutralHadronEnergyFraction) /* jet neutral hadron energy fraction
                                                     (relative to uncorrected jet energy) */ \
    JET_VAR(Float_t, neutralEmEnergyFraction) /* jet neutral EM energy fraction
                                                 (relative to uncorrected jet energy) */ \
    JET_VAR(Int_t, nConstituents) /* number of jet constituents */ \
    JET_VAR2(Int_t, chargedMultiplicity, neutralMultiplicity) /* jet charged and neutral multiplicities */ \
    JET_VAR2(Int_t, partonFlavour, hadronFlavour) /* parton-based and hadron-based flavours of the jet */ \
    JET_VAR(Float_t, m_softDrop) /* PUPPI soft-drop mass */ \
    JET_VAR4(Float_t, nJettiness_tau1, nJettiness_tau2, nJettiness_tau3, nJettiness_tau4) /* n-subjettiness variables
                                                                                             with PUPPI tau1-4 */ \
    JET_VAR4(std::vector<Float_t>, subJet_pt, subJet_eta, subJet_phi, subJet_mass) /* 4-momenta of sub-jets */ \
    /* Basic tau variables */ \
    TAU_VAR(Int_t, index) /* index of the tau */ \
    TAU_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the tau */ \
    TAU_VAR(Int_t, charge) /* tau charge */ \
    /* Tau ID variables */ \
    TAU_VAR(Int_t, decayMode) /* tau decay mode */ \
    TAU_VAR(Int_t, decayModeFinding) /* tau passed the old decay mode finding requirements */ \
    TAU_VAR(Int_t, decayModeFindingNewDMs) /* tau passed the new decay mode finding requirements */ \
    TAU_VAR(Float_t, chargedIsoPtSum) /* sum of the transverse momentums of charged pf candidates inside
                                         the tau isolation cone with dR < 0.5 */ \
    TAU_VAR(Float_t, chargedIsoPtSumdR03) /* sum of the transverse momentums of charged pf candidates inside
                                         the tau isolation cone with dR < 0.3 */ \
    TAU_VAR(Float_t, footprintCorrection) /* tau footprint correction inside the isolation cone with dR < 0.5 */ \
    TAU_VAR(Float_t, footprintCorrectiondR03) /* tau footprint correction inside the isolation cone with dR < 0.3 */ \
    TAU_VAR(Float_t, neutralIsoPtSum) /* sum of the transverse momentums of neutral pf candidates inside
                                         the tau isolation cone with dR < 0.5 */ \
    TAU_VAR(Float_t, neutralIsoPtSumWeight) /* weighted sum of the transverse momentums of neutral pf candidates inside
                                               the tau isolation cone with dR < 0.5 */ \
    TAU_VAR(Float_t, neutralIsoPtSumWeightdR03) /* weighted sum of the transverse momentums of neutral pf candidates
                                                   inside the tau isolation cone with dR < 0.3 */ \
    TAU_VAR(Float_t, neutralIsoPtSumdR03) /* sum of the transverse momentums of neutral pf candidates inside
                                             the tau isolation cone with dR < 0.3 */ \
    TAU_VAR(Float_t, photonPtSumOutsideSignalCone) /* sum of the transverse momentums of photons
                                                      inside the tau isolation cone with dR < 0.5 */ \
    TAU_VAR(Float_t, photonPtSumOutsideSignalConedR03) /* sum of the transverse momentums of photons inside
                                                          the tau isolation cone with dR < 0.3 */ \
    TAU_VAR(Float_t, puCorrPtSum) /* pile-up correction for the sum of the transverse momentums */ \
    TAU_IDS() \
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
    CAND_VAR(Int_t, pvAssociationQuality) /* information about how the association to the PV is obtained:
                                             NotReconstructedPrimary = 0, OtherDeltaZ = 1, CompatibilityBTag = 4,
                                             CompatibilityDz = 5, UsedInFitLoose = 6, UsedInFitTight = 7 */ \
    CAND_VAR(Int_t, fromPV) /* the association to PV=ipv. >=PVLoose corresponds to JME definition,
                               >=PVTight to isolation definition:
                               NoPV = 0, PVLoose = 1, PVTight = 2, PVUsedInFit = 3 */ \
    CAND_VAR(Float_t, puppiWeight) /* weight from full PUPPI */ \
    CAND_VAR(Float_t, puppiWeightNoLep) /* weight from PUPPI removing leptons */ \
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
    /* PAT electrons */ \
    ELE_VAR(Int_t, index) /* index of the electron */ \
    ELE_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the electron */ \
    ELE_VAR(Float_t, cc_ele_energy) /* energy of the first calo cluster in the electron super cluster */ \
    ELE_VAR(Float_t, cc_gamma_energy) /* sum of the energies of additional calo clusters
                                         in the electron super cluster */ \
    ELE_VAR(Int_t, cc_n_gamma) /* number of additional calo clusters in the electron super cluster */ \
    ELE_VAR3(Float_t, dxy, dxy_error, ip3d) /* impact parameter of the electron */ \
    ELE_VAR(Float_t, trackMomentumAtVtx) /* module of the track momentum at the PCA to the beam spot */ \
    ELE_VAR(Float_t, trackMomentumAtCalo) /* module of the track momentum extrapolated at the supercluster position
                                             from the innermost track state */ \
    ELE_VAR(Float_t, trackMomentumOut) /* module of the track momentum extrapolated at the seed cluster position
                                          from the outermost track state */ \
    ELE_VAR(Float_t, trackMomentumAtEleClus) /* module of the track momentum extrapolated at the ele cluster position
                                                from the outermost track state */ \
    ELE_VAR(Float_t, trackMomentumAtVtxWithConstraint) /* module of the track momentum at the PCA to the beam spot
                                                          using bs constraint */ \
    ELE_VAR(Float_t, ecalEnergy) /*  corrected ECAL energy */ \
    ELE_VAR(Float_t, ecalEnergy_error) /* uncertanty of the ECAL energy measurement */ \
    ELE_VAR(Float_t, eSuperClusterOverP) /* supercluster energy / track momentum at the PCA to the beam spot */ \
    ELE_VAR(Float_t, eSeedClusterOverP) /* seed cluster energy / track momentum at the PCA to the beam spot */ \
    ELE_VAR(Float_t, eSeedClusterOverPout) /* seed cluster energy / track momentum at calo extrapolated
                                              from the outermost track state */ \
    ELE_VAR(Float_t, eEleClusterOverPout) /* electron cluster energy / track momentum at calo extrapolated
                                             from the outermost track state */ \
    ELE_VAR(Float_t, deltaEtaSuperClusterTrackAtVtx) /* supercluster eta - track eta position at calo extrapolated
                                                        from innermost track state */ \
    ELE_VAR(Float_t, deltaEtaSeedClusterTrackAtCalo) /* seed cluster eta - track eta position at calo extrapolated
                                                        from the outermost track state */ \
    ELE_VAR(Float_t, deltaEtaEleClusterTrackAtCalo) /* electron cluster eta - track eta position at calo extrapolated
                                                       from the outermost state */ \
    ELE_VAR(Float_t, deltaEtaSeedClusterTrackAtVtx) /* cluster eta - track eta position at calo extrapolated
                                                       from innermost track state */ \
    ELE_VAR(Float_t, deltaPhiEleClusterTrackAtCalo) /* electron cluster phi - track phi position at calo extrapolated
                                                       from the outermost track state */ \
    ELE_VAR(Float_t, deltaPhiSuperClusterTrackAtVtx) /* supercluster phi - track phi position at calo extrapolated
                                                        from the innermost track state */ \
    ELE_VAR(Float_t, deltaPhiSeedClusterTrackAtCalo) /* seed cluster phi - track phi position at calo extrapolated
                                                        from the outermost track state */ \
    ELE_VAR2(Int_t, mvaInput_earlyBrem, mvaInput_lateBrem) /* early/late bremsstrahlung is detected:
                                                              unknown = -2, could not be evaluated = -1,
                                                              wrong = 0, true = 1 */ \
    ELE_VAR(Float_t, mvaInput_sigmaEtaEta) /* Sigma-eta-eta with the PF cluster */ \
    ELE_VAR(Float_t, mvaInput_hadEnergy) /* Associated PF Had Cluster energy */ \
    ELE_VAR(Float_t, mvaInput_deltaEta) /* PF-cluster GSF track delta-eta */ \
    ELE_VAR(Float_t, gsfTrack_normalizedChi2) /* chi^2 divided by number of degrees of freedom of the GSF track */ \
    ELE_VAR(Int_t, gsfTrack_numberOfValidHits) /* number of valid hits on the GSF track */ \
    ELE_VAR(Float_t, gsfTrack_pt) /* pt of the GSF track */ \
    ELE_VAR(Float_t, gsfTrack_pt_error) /* uncertainty of the pt measurement of the GSF track */ \
    ELE_VAR(Float_t, closestCtfTrack_normalizedChi2) /* chi^2 divided by number of degrees of freedom
                                                        of the closest CTF track */ \
    ELE_VAR(Int_t, closestCtfTrack_numberOfValidHits) /* number of valid hits on the closest CTF track */ \
    ELE_VAR4(Float_t, sigmaEtaEta, sigmaIetaIeta, sigmaIphiIphi, sigmaIetaIphi) /* shower shape spread */ \
    ELE_VAR4(Float_t, e1x5, e2x5Max, e5x5, r9) /* shower shape signatures */ \
    ELE_VAR4(Float_t, hcalDepth1OverEcal, hcalDepth2OverEcal, hcalDepth1OverEcalBc, hcalDepth2OverEcalBc) /* HCAL over ECAL variables */ \
    ELE_VAR4(Float_t, eLeft, eRight, eBottom, eTop) /* shower shape energies of cluster cell neighbours(?) */ \
    ELE_VAR4(Float_t, full5x5_sigmaEtaEta, full5x5_sigmaIetaIeta, full5x5_sigmaIphiIphi, full5x5_sigmaIetaIphi) /* shower shape spread */ \
    ELE_VAR4(Float_t, full5x5_e1x5, full5x5_e2x5Max, full5x5_e5x5, full5x5_r9) /* shower shape signatures */ \
    ELE_VAR4(Float_t, full5x5_hcalDepth1OverEcal, full5x5_hcalDepth2OverEcal, full5x5_hcalDepth1OverEcalBc, full5x5_hcalDepth2OverEcalBc) /* HCAL over ECAL variables */ \
    ELE_VAR4(Float_t, full5x5_eLeft, full5x5_eRight, full5x5_eBottom, full5x5_eTop) /* shower shape energies of cluster cell neighbours(?) */ \
    ELE_VAR4(Float_t, full5x5_e2x5Left, full5x5_e2x5Right, full5x5_e2x5Bottom, full5x5_e2x5Top) /* 2x5 shower shape energies of cluster cell neighbours(?) */ \
    /* Phase2 specific electron */ \
    ELE_VAR4(Float_t, hgcal_sigmaUU, hgcal_sigmaVV, hgcal_sigmaEE, hgcal_sigmaPP) /* HGCal cluster spread */ \
    ELE_VAR(Int_t, hgcal_nLayers) /* number of layers of HGCal cluster */ \
    ELE_VAR2(Int_t, hgcal_firstLayer, hgcal_lastLayer) /* first/last layer of HGCal cluster */ \
    ELE_VAR2(Int_t, hgcal_layerEfrac10, hgcal_layerEfrac90) /* layers with energy fraction of HGCal cluster */ \
    ELE_VAR(Float_t, hgcal_e4oEtot) /* e4 energy over total energy of the HGCal cluster (?) */ \
    ELE_VAR4(Float_t, hgcal_ecEnergy, hgcal_ecEnergyEE, hgcal_ecEnergyFH, hgcal_ecEnergyBH) /* energy in a cylinder around electron in HGCal */ \
    ELE_VAR(Float_t, hgcal_ecEt) /* transverse energy in a cylinder around electron shower axis in HGCal */ \
    ELE_VAR2(Float_t, hgcal_ecOrigEnergy, hgcal_ecOrigEt) /* original energy (including transverse) in a cylinder around electron shower axis in HGCal */ \
    ELE_VAR5(Float_t, hgcal_caloIsoRing0, hgcal_caloIsoRing1, hgcal_caloIsoRing2, hgcal_caloIsoRing3, hgcal_caloIsoRing4) /* Isolation sum rings around electron in HGCal */ \
    ELE_VAR4(Float_t, hgcal_depthCompatibility, hgcal_expectedDepth, hgcal_expectedSigma, hgcal_measuredDepth) /* shower depth variables for electron in HGCal */ \
    ELE_VAR3(Float_t, hgcal_pcaAxisX, hgcal_pcaAxisY, hgcal_pcaAxisZ) /* pca on axis X,Y,Z (?) */ \
    ELE_VAR3(Float_t, hgcal_pcaPositionX, hgcal_pcaPositionY, hgcal_pcaPositionZ) /* pca position in X,Y,Z (?) */ \
    ELE_VAR3(Float_t, hgcal_pcaEig1, hgcal_pcaEig2, hgcal_pcaEig3) /* pca eigenvalues (?) */ \
    ELE_VAR3(Float_t, hgcal_pcaSig1, hgcal_pcaSig2, hgcal_pcaSig3) /* pca significances (?) */ \
    /* PAT muons */ \
    MUON_VAR(Int_t, index) /* index of the muon */ \
    MUON_VAR4(Float_t, pt, eta, phi, mass) /* 4-momentum of the muon */ \
    MUON_VAR(Float_t, dxy) /* signed transverse impact parameter of the inner track wrt to the primary vertex */ \
    MUON_VAR(Float_t, dxy_error) /* uncertainty of the transverse impact parameter measurement */ \
    MUON_VAR(Float_t, normalizedChi2) /* chi^2 divided by number of degrees of freedom of the global track */ \
    MUON_VAR(Int_t, numberOfValidHits) /* number of valid hits on the global track */ \
    MUON_VAR(Float_t, segmentCompatibility) /* segment compatibility for a track with matched muon info */ \
    MUON_VAR(Float_t, caloCompatibility) /* relative likelihood based on ECAL, HCAL, HO energy defined as
                                            L_muon / (L_muon + L_not_muon) */ \
    MUON_VAR(Float_t, pfEcalEnergy) /* PF based energy deposition in the ECAL */ \
    MUON_VAR(UInt_t, type) /* Muon type - type of the algorithm that reconstructed this muon
                              multiple algorithms can reconstruct the same muon
                              GlobalMuon = 1 << 1;
                              TrackerMuon = 1 << 2;
                              StandAloneMuon = 1 << 3;
                              CaloMuon = 1 << 4;
                              PFMuon = 1 << 5;
                              RPCMuon = 1 << 6;
                              GEMMuon = 1 << 7;
                              ME0Muon = 1 << 8; */ \
    MUON_VAR(UInt_t, id) /* Muon ID: Loose = 1 << 0, Medium = 1 << 1, Tight = 1 << 2 */ \
    MUON_VAR(Float_t, pfRelIso04) /* PF-based relative isolation computed with 0.4 deltaR cone */ \
    MUON_VAR4(Int_t, n_matches_DT_1, n_matches_DT_2, n_matches_DT_3, \
                     n_matches_DT_4) /* number of segment matches for the DT subdetector stations */ \
    MUON_VAR4(Int_t, n_matches_CSC_1, n_matches_CSC_2, n_matches_CSC_3, \
                     n_matches_CSC_4) /* number of segment matches for the CSC subdetector stations */ \
    MUON_VAR4(Int_t, n_matches_RPC_1, n_matches_RPC_2, n_matches_RPC_3, \
                     n_matches_RPC_4) /* number of segment matches for the RPC subdetector stations */ \
    MUON_VAR4(Int_t, n_matches_GEM_1, n_matches_GEM_2, n_matches_GEM_3, \
                     n_matches_GEM_4) /* number of segment matches for the GEM subdetector stations */ \
    MUON_VAR4(Int_t, n_matches_ME0_1, n_matches_ME0_2, n_matches_ME0_3, \
                     n_matches_ME0_4) /* number of segment matches for the ME0 subdetector stations */ \
    MUON_VAR4(Int_t, n_hits_DT_1, n_hits_DT_2, n_hits_DT_3, \
                     n_hits_DT_4) /* number of valid and bad hits for the DT subdetector stations */ \
    MUON_VAR4(Int_t, n_hits_CSC_1, n_hits_CSC_2, n_hits_CSC_3, \
                     n_hits_CSC_4) /* number of valid and bad hits for the CSC subdetector stations */ \
    MUON_VAR4(Int_t, n_hits_RPC_1, n_hits_RPC_2, n_hits_RPC_3, \
                     n_hits_RPC_4) /* number of valid and bad hits for the RPC subdetector stations */ \
    MUON_VAR4(Int_t, n_hits_GEM_1, n_hits_GEM_2, n_hits_GEM_3, \
                     n_hits_GEM_4) /* number of valid and bad hits for the GEM subdetector stations */ \
    MUON_VAR4(Int_t, n_hits_ME0_1, n_hits_ME0_2, n_hits_ME0_3, \
                     n_hits_ME0_4) /* number of valid and bad hits for the ME0 subdetector stations */ \
    /* Isolated tracks */ \
    TRACK_VAR(Int_t, index) /* index of the track */ \
    TRACK_VAR3(Float_t, pt, eta, phi) /* track kinematics */ \
    TRACK_VAR(Int_t, fromPV) /* the association to PV=ipv. >=PVLoose corresponds to JME definition,
                               >=PVTight to isolation definition:
                               NoPV = 0, PVLoose = 1, PVTight = 2, PVUsedInFit = 3 */ \
    TRACK_VAR(Int_t, charge) /* electric charge */ \
    TRACK_VAR(Float_t, dxy) /* signed transverse impact parameter wrt to the primary vertex */ \
    TRACK_VAR(Float_t, dxy_error) /* uncertainty of the transverse impact parameter measurement */ \
    TRACK_VAR(Float_t, dz) /* dz wrt to the primary vertex */ \
    TRACK_VAR(Float_t, dz_error) /* uncertainty of the dz measurement */ \
    TRACK_VAR3(Int_t, isHighPurityTrack, isTightTrack, isLooseTrack) /* track Quality */ \
    TRACK_VAR(Float_t, dEdxStrip) /* estimated dE/dx values in the strips */ \
    TRACK_VAR(Float_t, dEdxPixel) /*  estimated dE/dx values in the pixels */ \
    TRACK_VAR2(Float_t, deltaEta, deltaPhi) /* the difference in eta/phi between the initial track trajectory and
                                               the point of intersection with the Ecal.  Can be used to identify
                                               roughly the calorimeter cells the track should hit.*/ \
    TRACK_VAR(Int_t, n_ValidHits) /* number Of Valid Hits */ \
    TRACK_VAR(Int_t, n_BadHits) /* number Of Bad Hits */ \
    TRACK_VAR3(Int_t, n_TimingHits, n_ValidTimingHits, n_LostTimingHits) /* number Of Timing Hits */ \
    TRACK_VAR4(Int_t, n_MuonHits, n_ValidMuonHits, n_LostMuonHits, n_BadMuonHits) /* number Of Muon Hits */ \
    TRACK_VAR3(Int_t, n_ValidMuonDTHits, n_LostMuonDTHits, n_BadMuonDTHits) /* number Of Muon DT Hits */ \
    TRACK_VAR3(Int_t, n_ValidMuonCSCHits, n_LostMuonCSCHits, n_BadMuonCSCHits) /* number Of Muon CSC Hits */ \
    TRACK_VAR3(Int_t, n_ValidMuonRPCHits, n_LostMuonRPCHits, n_BadMuonRPCHits) /* number Of Muon RPC Hits */ \
    TRACK_VAR3(Int_t, n_ValidMuonGEMHits, n_LostMuonGEMHits, n_BadMuonGEMHits) /* number Of Muon GEM Hits */ \
    TRACK_VAR3(Int_t, n_ValidMuonME0Hits, n_LostMuonME0Hits, n_BadMuonME0Hits) /* number Of Muon ME0 Hits */ \
    TRACK_VAR(Int_t, n_InactiveHits) /* number Of Inactive Hits */ \
    TRACK_VAR3(Int_t, n_AllHits_TRACK, n_AllHits_MISSING_INNER, n_AllHits_MISSING_OUTER) /* number Of All Hits */ \
    TRACK_VAR3(Int_t, n_LostHits_TRACK, n_LostHits_MISSING_INNER, n_LostHits_MISSING_OUTER) /* number Of Lost Hits */ \
    TRACK_VAR(Int_t, n_ValidPixelHits) /* number Of Valid Pixel Hits */ \
    TRACK_VAR(Int_t, n_ValidStripHits) /* number Of Valid Strip Hits */ \
    TRACK_VAR3(Int_t, n_LostPixelHits_TRACK, n_LostPixelHits_MISSING_INNER, \
                      n_LostPixelHits_MISSING_OUTER) /* number Of Lost Pixel Hits */ \
    TRACK_VAR3(Int_t, n_LostStripHits_TRACK, n_LostStripHits_MISSING_INNER, \
                      n_LostStripHits_MISSING_OUTER) /* number Of Lost Strip Hits */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, Tau, TauTuple, TAU_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, TauTuple, TAU_DATA)
#undef VAR
#undef VAR2
#undef VAR3
#undef VAR4
#undef TAU_DATA
#undef CAND_VAR
#undef CAND_VAR2
#undef CAND_VAR3
#undef CAND_VAR4
#undef ELE_VAR
#undef ELE_VAR2
#undef ELE_VAR3
#undef ELE_VAR4
#undef MUON_VAR
#undef MUON_VAR2
#undef MUON_VAR3
#undef MUON_VAR4
#undef TRACK_VAR
#undef TRACK_VAR2
#undef TRACK_VAR3
#undef TRACK_VAR4
#undef TAU_ID

namespace tau_tuple {

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
