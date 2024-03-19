
import ROOT
ROOT.gInterpreter.Declare("""

using RVecF = ROOT::VecOps::RVec<float>;
using RVecB = ROOT::VecOps::RVec<bool>;

template<typename T>
T DeltaR(T eta1, T phi1, T eta2, T phi2){
  const T dphi = ROOT::Math::VectorUtil::Phi_mpi_pi(phi2 - phi1);
  const T deta = eta2-eta1;
  return std::hypot(dphi, deta);
}

RVecB DeltaRMatch(const RVecF& eta1, const RVecF& phi1,
                  const std::vector<const RVecF*>& eta2,
                  const std::vector<const RVecF*>& phi2, float dR)
{
  RVecB mask(eta1.size(), false);
  for(size_t idx1 = 0; idx1 < eta1.size(); ++idx1) {
    bool has_match = false;
    for(size_t col = 0; !has_match && col < eta2.size(); ++col) {
      for(size_t idx2 = 0; !has_match && idx2 < eta2[col]->size(); ++idx2) {
        has_match = DeltaR(eta1[idx1], phi1[idx1], eta2[col]->at(idx2), phi2[col]->at(idx2)) < dR;
      }
    }
    mask[idx1] = has_match;
  }
  return mask;
}
""")

def skim(df):
  columns = [ str(c) for c in df.GetColumnNames() ]
  ref_objects = [ ('L1Tau', '_tmp'), ('Tau', ''), ('Jet', '') ]
  for obj, suffix in ref_objects:
    df = df.Define(f'{obj}_sel', f'{obj}_pt > 15 && abs({obj}_eta) < 2.7')
    sample_c = None
    for c in columns:
      if c.startswith(obj + '_'):
        if len(suffix) > 0:
          df = df.Define(c + suffix, f'{c}[{obj}_sel]')
        else:
          df = df.Redefine(c, f'{c}[{obj}_sel]')
        sample_c = c
    counter = 'n' + obj
    if len(suffix) == 0 and sample_c and counter in columns:
        df = df.Redefine(counter, f'static_cast<int>({sample_c}.size())')
  ref_eta = [ f'&{obj}_eta{suffix}' for obj, suffix in ref_objects ]
  ref_phi = [ f'&{obj}_phi{suffix}' for obj, suffix in ref_objects ]
  ref_eta_str = '{' + ', '.join(ref_eta) + '}'
  ref_phi_str = '{' + ', '.join(ref_phi) + '}'
  for obj in [ 'PixelTrack', 'PFCand', 'RecHitHBHE', 'RecHitEB', 'RecHitEE' ]:
    if f'{obj}_eta' not in columns:
      continue
    df = df.Define(f'{obj}_sel', f'DeltaRMatch({obj}_eta, {obj}_phi, {ref_eta_str}, {ref_phi_str}, 0.5f)')
    sample_c = None
    for c in columns:
      if c.startswith(obj + '_'):
        df = df.Redefine(c, f'{c}[{obj}_sel]')
        sample_c = c
    counter = 'n' + obj
    if sample_c and counter in columns:
        df = df.Redefine(counter, f'static_cast<int>({sample_c}.size())')
  return df