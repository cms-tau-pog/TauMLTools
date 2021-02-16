struct setup {
  static constexpr size_t    n_tau              = 500;
  static constexpr Long64_t  start_dataset      = 0;
  static constexpr Long64_t  end_dataset        = std::numeric_limits<Long64_t>::max();
  static constexpr size_t    n_inner_cells      = 11;  // number of inner cells in eta and phi
  static constexpr Double_t  inner_cell_size    = 0.02; // size of the inner cell in eta and phi
  static constexpr size_t    n_outer_cells      = 21; // number of outer cells in eta and phi
  static constexpr Double_t  outer_cell_size    = 0.05; // size of the outer cell in eta and phi
  static constexpr Int_t     n_threads          = 1; // number of threads
  static constexpr size_t    n_fe_tau    = 43;  // number of high level featurese of tau
  static constexpr size_t    n_pf_el     = 22; // Number of features for PfCand_electron
  static constexpr size_t    n_pf_mu     = 23; // Number of features for PfCand_electron
  static constexpr size_t    n_pf_chHad  = 27;
  static constexpr size_t    n_pf_nHad   = 7;
  static constexpr size_t    n_pf_gamma  = 23;
  static constexpr size_t    n_ele       = 37;
  static constexpr size_t    n_muon      = 37;
  static constexpr Int_t     tau_types   = 4; /*
                                                tau_e       = "tauType==0"
                                                tau_mu      = "tauType==1"
                                                tau_h       = "tauType==2"
                                                tau_jet     = "tauType==3"
                                              */
  inline static const std::vector<std::string> input_dirs{"/eos/cms/store/group/phys_tau/TauML/prod_2018_v1/ShuffleMergeSpectral_v0/"};
  inline static const std::string file_name_pattern = "^.*_(1|2|3|4|5|6|7|8|9|10).root$"; // As a test take 1-10 files
  // inline static const std::string file_name_pattern = "^.*_.*.root$";
  inline static const std::string exclude_list = "";
  inline static const std::string exclude_dir_list = "";
};

enum class CellObjectType {
  PfCand_electron,
  PfCand_muon,
  PfCand_chargedHadron,
  PfCand_neutralHadron,
  PfCand_gamma,
  Electron,
  Muon
};

enum class TauFlat_f {
  tau_pt = 0,
  tau_eta = 1,
  tau_phi = 2,
  tau_mass = 3,
  tau_E_over_pt = 4,
  tau_charge = 5,
  tau_n_charged_prongs = 6,
  tau_n_neutral_prongs = 7,
  chargedIsoPtSum = 8,
  chargedIsoPtSumdR03_over_dR05 = 9,
  footprintCorrection = 10,
  neutralIsoPtSum = 11,
  neutralIsoPtSumWeight_over_neutralIsoPtSum = 12,
  neutralIsoPtSumWeightdR03_over_neutralIsoPtSum = 13,
  neutralIsoPtSumdR03_over_dR05 = 14,
  photonPtSumOutsideSignalCone = 15,
  puCorrPtSum = 16,
  tau_dxy_valid = 17,
  tau_dxy = 18,
  tau_dxy_sig = 19,
  tau_ip3d_valid = 20,
  tau_ip3d = 21,
  tau_ip3d_sig = 22,
  tau_dz = 23,
  tau_dz_sig_valid = 24,
  tau_dz_sig = 25,
  tau_flightLength_x = 26,
  tau_flightLength_y = 27,
  tau_flightLength_z = 28,
  tau_flightLength_sig = 29,
  tau_pt_weighted_deta_strip = 30,
  tau_pt_weighted_dphi_strip = 31,
  tau_pt_weighted_dr_signal = 32,
  tau_pt_weighted_dr_iso = 33,
  tau_leadingTrackNormChi2 = 34,
  tau_e_ratio_valid = 35,
  tau_e_ratio = 36,
  tau_gj_angle_diff_valid = 37,
  tau_gj_angle_diff = 38,
  tau_n_photons = 39,
  tau_emFraction = 40,
  tau_inside_ecal_crack = 41,
  leadChargedCand_etaAtEcalEntrance_minus_tau_eta = 42
};

enum class PfCand_electron_f {
  pfCand_ele_valid = 0,
  pfCand_ele_rel_pt = 1,
  pfCand_ele_deta = 2,
  pfCand_ele_dphi = 3,
  pfCand_ele_pvAssociationQuality = 4,
  pfCand_ele_puppiWeight = 5,
  pfCand_ele_charge = 6,
  pfCand_ele_lostInnerHits = 7,
  pfCand_ele_numberOfPixelHits = 8,
  pfCand_ele_vertex_dx = 9,
  pfCand_ele_vertex_dy = 10,
  pfCand_ele_vertex_dz = 11,
  pfCand_ele_vertex_dx_tauFL = 12,
  pfCand_ele_vertex_dy_tauFL = 13,
  pfCand_ele_vertex_dz_tauFL = 14,
  pfCand_ele_hasTrackDetails = 15,
  pfCand_ele_dxy = 16,
  pfCand_ele_dxy_sig = 17,
  pfCand_ele_dz = 18,
  pfCand_ele_dz_sig = 19,
  pfCand_ele_track_chi2_ndof = 20,
  pfCand_ele_track_ndof = 21
};

enum class PfCand_muon_f {
  pfCand_muon_valid = 0,
  pfCand_muon_rel_pt = 1,
  pfCand_muon_deta = 2,
  pfCand_muon_dphi = 3,
  pfCand_muon_pvAssociationQuality = 4,
  pfCand_muon_fromPV = 5,
  pfCand_muon_puppiWeight = 6,
  pfCand_muon_charge = 7,
  pfCand_muon_lostInnerHits = 8,
  pfCand_muon_numberOfPixelHits = 9,
  pfCand_muon_vertex_dx = 10,
  pfCand_muon_vertex_dy = 11,
  pfCand_muon_vertex_dz = 12,
  pfCand_muon_vertex_dx_tauFL = 13,
  pfCand_muon_vertex_dy_tauFL = 14,
  pfCand_muon_vertex_dz_tauFL = 15,
  pfCand_muon_hasTrackDetails = 16,
  pfCand_muon_dxy = 17,
  pfCand_muon_dxy_sig = 18,
  pfCand_muon_dz = 19,
  pfCand_muon_dz_sig = 20,
  pfCand_muon_track_chi2_ndof = 21,
  pfCand_muon_track_ndof = 22,

};

enum class PfCand_chargedHadron_f {
  pfCand_chHad_valid  = 0,
  pfCand_chHad_rel_pt = 1,
  pfCand_chHad_deta = 2,
  pfCand_chHad_dphi = 3,
  pfCand_chHad_leadChargedHadrCand = 4,
  pfCand_chHad_pvAssociationQuality = 5,
  pfCand_chHad_fromPV = 6,
  pfCand_chHad_puppiWeight = 7,
  pfCand_chHad_puppiWeightNoLep = 8,
  pfCand_chHad_charge = 9,
  pfCand_chHad_lostInnerHits = 10,
  pfCand_chHad_numberOfPixelHits = 11,
  pfCand_chHad_vertex_dx = 12,
  pfCand_chHad_vertex_dy = 13,
  pfCand_chHad_vertex_dz = 14,
  pfCand_chHad_vertex_dx_tauFL = 15,
  pfCand_chHad_vertex_dy_tauFL = 16,
  pfCand_chHad_vertex_dz_tauFL = 17,
  pfCand_chHad_hasTrackDetails = 18,
  pfCand_chHad_dxy = 19,
  pfCand_chHad_dxy_sig = 20,
  pfCand_chHad_dz = 21,
  pfCand_chHad_dz_sig = 22,
  pfCand_chHad_track_chi2_ndof = 23,
  pfCand_chHad_track_ndof = 24,
  pfCand_chHad_hcalFraction = 25,
  pfCand_chHad_rawCaloFraction = 26
};

enum class PfCand_neutralHadron_f {
  pfCand_nHad_valid = 0,
  pfCand_nHad_rel_pt = 1,
  pfCand_nHad_deta = 2,
  pfCand_nHad_dphi = 3,
  pfCand_nHad_puppiWeight = 4,
  pfCand_nHad_puppiWeightNoLep = 5,
  pfCand_nHad_hcalFraction = 6

};

enum class pfCand_gamma_f {
  pfCand_gamma_valid  = 0,
  pfCand_gamma_rel_pt = 1,
  pfCand_gamma_deta = 2,
  pfCand_gamma_dphi = 3,
  pfCand_gamma_pvAssociationQuality = 4,
  pfCand_gamma_fromPV = 5,
  pfCand_gamma_puppiWeight = 6,
  pfCand_gamma_puppiWeightNoLep = 7,
  pfCand_gamma_lostInnerHits = 8,
  pfCand_gamma_numberOfPixelHits = 9,
  pfCand_gamma_vertex_dx = 10,
  pfCand_gamma_vertex_dy = 11,
  pfCand_gamma_vertex_dz = 12,
  pfCand_gamma_vertex_dx_tauFL = 13,
  pfCand_gamma_vertex_dy_tauFL = 14,
  pfCand_gamma_vertex_dz_tauFL = 15,
  pfCand_gamma_hasTrackDetails = 16,
  pfCand_gamma_dxy = 17,
  pfCand_gamma_dxy_sig = 18,
  pfCand_gamma_dz = 19,
  pfCand_gamma_dz_sig = 20,
  pfCand_gamma_track_chi2_ndof = 21,
  pfCand_gamma_track_ndof = 22
};

enum class Electron_f {
  ele_valid = 0,
  ele_rel_pt = 1,
  ele_deta = 2,
  ele_dphi = 3,
  ele_cc_valid = 4,
  ele_cc_ele_rel_energy = 5,
  ele_cc_gamma_rel_energy = 6,
  ele_cc_n_gamma = 7,
  ele_rel_trackMomentumAtVtx = 8,
  ele_rel_trackMomentumAtCalo = 9,
  ele_rel_trackMomentumOut = 10,
  ele_rel_trackMomentumAtEleClus = 11,
  ele_rel_trackMomentumAtVtxWithConstraint = 12,
  ele_rel_ecalEnergy = 13,
  ele_ecalEnergy_sig = 14,
  ele_eSuperClusterOverP = 15,
  ele_eSeedClusterOverP = 16,
  ele_eSeedClusterOverPout = 17,
  ele_eEleClusterOverPout = 18,
  ele_deltaEtaSuperClusterTrackAtVtx = 19,
  ele_deltaEtaSeedClusterTrackAtCalo = 20,
  ele_deltaEtaEleClusterTrackAtCalo = 21,
  ele_deltaPhiEleClusterTrackAtCalo = 22,
  ele_deltaPhiSuperClusterTrackAtVtx = 23,
  ele_deltaPhiSeedClusterTrackAtCalo = 24,
  ele_mvaInput_earlyBrem = 25,
  ele_mvaInput_lateBrem = 26,
  ele_mvaInput_sigmaEtaEta = 27,
  ele_mvaInput_hadEnergy = 28,
  ele_mvaInput_deltaEta = 29,
  ele_gsfTrack_normalizedChi2 = 30,
  ele_gsfTrack_numberOfValidHits = 31,
  ele_rel_gsfTrack_pt = 32,
  ele_gsfTrack_pt_sig = 33,
  ele_has_closestCtfTrack = 34,
  ele_closestCtfTrack_normalizedChi2 = 35,
  ele_closestCtfTrack_numberOfValidHits = 36
};

enum class Muon_f {
  muon_valid = 0,
  muon_rel_pt = 1,
  muon_deta = 2,
  muon_dphi = 3,
  muon_dxy = 4,
  muon_dxy_sig = 5,
  muon_normalizedChi2_valid = 6,
  muon_normalizedChi2 = 7,
  muon_numberOfValidHits = 8,
  muon_segmentCompatibility = 9,
  muon_caloCompatibility = 10,
  muon_pfEcalEnergy_valid = 11,
  muon_rel_pfEcalEnergy = 12,
  muon_n_matches_DT_1 = 13,
  muon_n_matches_DT_2 = 14,
  muon_n_matches_DT_3 = 15,
  muon_n_matches_DT_4 = 16,
  muon_n_matches_CSC_1 = 17,
  muon_n_matches_CSC_2 = 18,
  muon_n_matches_CSC_3 = 19,
  muon_n_matches_CSC_4 = 20,
  muon_n_matches_RPC_1 = 21,
  muon_n_matches_RPC_2 = 22,
  muon_n_matches_RPC_3 = 23,
  muon_n_matches_RPC_4 = 24,
  muon_n_hits_DT_1 = 25,
  muon_n_hits_DT_2 = 26,
  muon_n_hits_DT_3 = 27,
  muon_n_hits_DT_4 = 28,
  muon_n_hits_CSC_1 = 29,
  muon_n_hits_CSC_2 = 30,
  muon_n_hits_CSC_3 = 31,
  muon_n_hits_CSC_4 = 32,
  muon_n_hits_RPC_1 = 33,
  muon_n_hits_RPC_2 = 34,
  muon_n_hits_RPC_3 = 35,
  muon_n_hits_RPC_4 = 36
};
