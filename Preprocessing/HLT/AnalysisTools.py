import datetime
from enum import Enum
import ROOT
import os

class TauType(Enum):
  other = -1
  e = 0
  mu = 1
  tau = 2
  jet = 3
  emb_e = 4
  emb_mu = 5
  emb_tau = 6
  emb_jet = 7
  data = 8
  displaced_e = 9
  displaced_mu = 10
  displaced_tau = 11
  displaced_jet = 12

def ListToVector(list, type="string"):
	vec = ROOT.std.vector(type)()
	for item in list:
		vec.push_back(item)
	return vec

def GenerateEnumCppString(cls):
  enum_string = f"enum class {cls.__name__} : int {{\n"
  for item in cls:
    enum_string += f"    {item.name} = {item.value},\n"
  enum_string += "};"
  return enum_string

def PrepareRootEnv():
  headers_dir = os.path.dirname(os.path.abspath(__file__))
  enums = [ TauType ]
  headers = [ 'AnalysisTools.h', 'GenLepton.h', 'GenTools.h', 'TupleMaker.h' ]
  for cls in enums:
    cls_str = GenerateEnumCppString(cls)
    if not ROOT.gInterpreter.Declare(cls_str):
      raise RuntimeError(f'Failed to declare {cls.__name__}')
  for header in headers:
    header_path = os.path.join(headers_dir, header)
    if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
      raise RuntimeError(f'Failed to load {header_path}')

def timestamp_str():
  t = datetime.datetime.now()
  t_str = t.strftime('%Y-%m-%d %H:%M:%S')
  return f'[{t_str}] '

def MakeFileList(input):
  file_list = []
  if isinstance(input, str):
    input = input.split(',')
  for item in input:
    if not os.path.exists(item):
      raise RuntimeError(f'File or directory {item} does not exist')
    if os.path.isdir(item):
      for root, dirs, files in os.walk(item):
        for file in files:
          if file.endswith('.root'):
            file_list.append(os.path.join(root, file))
    else:
      file_list.append(item)
  return sorted(file_list)

def ApplyCommonDefinitions(df, deltaR=0.4, isData=False):
  df = df.Define("genLeptons", """reco_tau::gen_truth::GenLepton::fromNanoAOD(
                                    GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass,
                                    GenPart_genPartIdxMother, GenPart_pdgId, GenPart_statusFlags,
                                    GenPart_vx, GenPart_vy, GenPart_vz, event)""") \
         .Define('L1Tau_mass', 'RVecF(L1Tau_pt.size(), 0.)') \
         .Define('L1Tau_p4', 'GetP4(L1Tau_pt, L1Tau_eta, L1Tau_phi, L1Tau_mass)') \
         .Define('Jet_p4', 'GetP4(Jet_pt, Jet_eta, Jet_phi, Jet_mass)') \
         .Define('Tau_p4', 'GetP4(Tau_pt, Tau_eta, Tau_phi, Tau_mass)') \
         .Define('L1Tau_jetIdx', f'FindUniqueMatching(L1Tau_p4, Jet_p4, {deltaR})') \
         .Define('L1Tau_tauIdx', f'FindUniqueMatching(L1Tau_p4, Tau_p4, {deltaR})')

  if isData:
    df = df.Define('L1Tau_Gen_type', 'RVecI(L1Tau_pt.size(), static_cast<int>(TauType::data))')
  else:
    df = df.Define('GenJet_p4', 'GetP4(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass)') \
           .Define('GenLepton_p4', 'v_ops::visibleP4(genLeptons)') \
           .Define('L1Tau_genLepIndices', f'FindMatchingSet(L1Tau_p4, GenLepton_p4, {deltaR})') \
           .Define('L1Tau_genLepUniqueIdx', f'FindUniqueMatching(L1Tau_p4, GenLepton_p4, {deltaR})') \
           .Define('L1Tau_genJetUniqueIdx', f'FindUniqueMatching(L1Tau_p4, GenJet_p4, {deltaR})') \
           .Define('L1Tau_Gen_type', '''GetTauTypes(genLeptons, L1Tau_genLepUniqueIdx, L1Tau_genLepIndices,
                                                       L1Tau_genJetUniqueIdx, false, true)''') \
           .Define('L1Tau_Gen_p4', '''GetGenP4(L1Tau_Gen_type, L1Tau_genLepUniqueIdx, L1Tau_genJetUniqueIdx,
                                               genLeptons, GenJet_p4)''') \
           .Define('L1Tau_Gen_pt', 'v_ops::pt(L1Tau_Gen_p4)') \
           .Define('L1Tau_Gen_eta', 'v_ops::eta(L1Tau_Gen_p4)') \
           .Define('L1Tau_Gen_abs_eta', 'abs(L1Tau_Gen_eta)') \
           .Define('L1Tau_Gen_phi', 'v_ops::phi(L1Tau_Gen_p4)') \
           .Define('L1Tau_Gen_mass', 'v_ops::mass(L1Tau_Gen_p4)') \
           .Define('L1Tau_Gen_charge', 'GetGenCharge(L1Tau_Gen_type, L1Tau_genLepUniqueIdx, genLeptons)') \
           .Define('L1Tau_Gen_partonFlavour', '''GetGenPartonFlavour(L1Tau_Gen_type, L1Tau_genJetUniqueIdx,
                                                 GenJet_partonFlavour)''') \
           .Define('L1Tau_Gen_flightLength', 'GetGenFlightLength(L1Tau_Gen_type, L1Tau_genLepUniqueIdx, genLeptons)') \
           .Define('L1Tau_Gen_flightLength_rho', 'v_ops::rho(L1Tau_Gen_flightLength)') \
           .Define('L1Tau_Gen_flightLength_phi', 'v_ops::phi(L1Tau_Gen_flightLength)') \
           .Define('L1Tau_Gen_flightLength_z', 'v_ops::z(L1Tau_Gen_flightLength)')
  return df




