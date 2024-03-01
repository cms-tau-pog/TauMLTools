import os
import ROOT
ROOT.gROOT.SetBatch(True)

def ListToVector(list, type="string"):
	vec = ROOT.std.vector(type)()
	for item in list:
		vec.push_back(item)
	return vec

headers_dir = os.path.dirname(os.path.abspath(__file__))
for header in [ 'GenLepton.h' ]:
  header_path = os.path.join(headers_dir, header)
  if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
    raise RuntimeError(f'Failed to load {header_path}')


ROOT.gInterpreter.Declare('''
using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecS = ROOT::VecOps::RVec<size_t>;
using RVecUC = ROOT::VecOps::RVec<UChar_t>;
using RVecF = ROOT::VecOps::RVec<float>;
using RVecB = ROOT::VecOps::RVec<bool>;

template<typename T1, typename T2>
T1 DeltaPhi(T1 phi1, T2 phi2){
  return ROOT::Math::VectorUtil::Phi_mpi_pi(phi2 - phi1);
}
template<typename T1, typename T2>
T1 DeltaEta(T1 eta1, T2 eta2){
  return (eta2-eta1);
}

template<typename T1, typename T2>
T1 DeltaR(T1 eta1, T1 phi1, T2 eta2, T2 phi2){
  T1 dphi = DeltaPhi(phi1, phi2);
  T1 deta = DeltaEta(eta1, eta2);
  return std::hypot(dphi, deta);
}
''')

sample_path = "/eos/cms/store/group/phys_tau/kandroso/Run3_HLT/prod_v1/GluGluHToTauTau_M-125"
files = '*.root'
#files = 'nano_2.root'
df = ROOT.RDataFrame("Events", os.path.join(sample_path, files))

df = df.Define("genLeptons", """
  reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass,
                                              GenPart_genPartIdxMother, GenPart_pdgId, GenPart_statusFlags, event)
""")
df = df.Define('genTauIdx', '''
  RVecS indices;
  for(size_t n = 0; n < genLeptons.size(); ++n) {
    if(genLeptons[n].kind() == reco_tau::gen_truth::GenLepton::Kind::TauDecayedToHadrons
        && std::abs(genLeptons[n].visibleP4().eta()) < 2.1)
      indices.push_back(n);
  }
  return indices;
''')

df = df.Define(f'genTau_l1Match', '''
  RVecI result(genLeptons.size(), -1);
  for (size_t idx : genTauIdx) {
    int matched = -1;
    for(size_t n = 0; n < L1Tau_pt.size(); ++n) {
      const auto dR = DeltaR(genLeptons[idx].visibleP4().eta(), genLeptons[idx].visibleP4().phi(),
                            L1Tau_eta[n], L1Tau_phi[n]);
      if(dR < 0.4) {
        matched = n;
        break;
      }
    }
    result[idx] = matched;
  }
  return result;
''')

for var in [ 'pt', 'eta', 'phi', 'mass' ]:
  df = df.Define(f'genTau_{var}', f'''
    RVecF result;
    for(size_t idx : genTauIdx)
      result.push_back(genLeptons[idx].visibleP4().{var}());
    return result;
  ''')
  df = df.Define(f'genTau_{var}_matched', f'''
    RVecF result;
    for(size_t idx : genTauIdx) {{
      if (genTau_l1Match[idx] >= 0)
        result.push_back(genLeptons[idx].visibleP4().{var}());
    }}
    return result;
  ''')
  df = df.Define(f'genTau_{var}_matchedIso', f'''
    RVecF result;
    for(size_t idx : genTauIdx) {{
      if (genTau_l1Match[idx] >= 0 && L1Tau_hwIso[genTau_l1Match[idx]] > 0)
        result.push_back(genLeptons[idx].visibleP4().{var}());
    }}
    return result;
  ''')


df = df.Define('nGenLeptons', 'genLeptons.size()')

pt_bins = [
  10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500,
]
pt_bins = ListToVector(pt_bins, "double")
pt_model = ROOT.RDF.TH1DModel('', '', pt_bins.size() - 1, pt_bins.data())

hist = df.Histo1D(pt_model, 'genTau_pt')
hist_matched = df.Histo1D(pt_model, 'genTau_pt_matched')
hist_matched_iso = df.Histo1D(pt_model, 'genTau_pt_matchedIso')

eff = ROOT.TEfficiency(hist_matched.GetValue(), hist.GetValue())
eff.SetStatisticOption(ROOT.TEfficiency.kFCP)

eff_iso = ROOT.TEfficiency(hist_matched_iso.GetValue(), hist.GetValue())
eff_iso.SetStatisticOption(ROOT.TEfficiency.kFCP)

ROOT.gStyle.SetOptStat(0)
canvas = ROOT.TCanvas("canvas", "canvas", 800, 800)
canvas.Draw()
canvas.SetLogx()
hist_base = pt_model.GetHistogram()
hist_base.Draw()
hist_base.GetXaxis().SetTitle("visible gen #tau p_{T} (GeV)")
hist_base.GetXaxis().SetNoExponent(1)
hist_base.GetXaxis().SetMoreLogLabels(1)
hist_base.GetXaxis().SetTitleOffset(1.2)
hist_base.GetYaxis().SetTitle("efficiency")
hist_base.SetTitle("L1 reconstruction efficiency for #tau_{h}")
eff.Draw("SAME P")
eff_iso.SetLineColor(ROOT.kRed)
eff_iso.Draw("SAME P")
canvas.Update()
legend = ROOT.TLegend(0.6, 0.2, 0.9, 0.4)
legend.AddEntry(eff, "L1 #tau", "F")
legend.AddEntry(eff_iso, "iso L1 #tau", "F")
legend.Draw()

canvas.SaveAs("genTau_pt_eff.pdf")

