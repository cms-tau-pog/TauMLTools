import ROOT
import numpy as np
from array import array

myFile = ROOT.TFile('file:eventTuple.root')
myTree = myFile.Get('taus')

rdf = ROOT.RDataFrame(myTree)

# Use to run on a limited number of events
# rdf = rdf.Range(5000)

ROOT.gInterpreter.ProcessLine('''

#include "../interface/GenLepton.h"
#include "../interface/CPfunctions.h"

''')

rdf = rdf.Filter('genLepton_index >=0 && genLepton_kind == 5 && genLepton_vis_pt >= 40 && std::abs(genLepton_vis_eta) < 2.1')\
         .Define('genLepton','''reco_tau::gen_truth::GenLepton::fromRootTuple(
           genLepton_lastMotherIndex, genParticle_pdgId, genParticle_mother, genParticle_charge,
           genParticle_isFirstCopy, genParticle_isLastCopy,
           genParticle_pt, genParticle_eta, genParticle_phi, genParticle_mass,
           genParticle_vtx_x, genParticle_vtx_y, genParticle_vtx_z)
         ''')

rdf = rdf.Define('genLepton_mothers','genLepton.mothers()')

# tau->pi,rho,a1
rdf = rdf.Filter('(genLepton.nChargedHadrons() == 1 && genLepton.nNeutralHadrons() == 0) || (genLepton.nChargedHadrons() == 1 && genLepton.nNeutralHadrons() == 1) || (genLepton.nChargedHadrons() == 3 && genLepton.nNeutralHadrons() == 0)')\
         .Define('PhiCP','''tau_cp::CPfunctions::PhiCP(genLepton, evt)''')

rdf = rdf.Filter('PhiCP > 0')

canvas = ROOT.TCanvas()
stack = ROOT.THStack()

hist = rdf.Histo1D(("", "", 5, 0., 2*ROOT.TMath.Pi()), 'PhiCP')
hist.SetLineColor(3)
hist.Draw()

canvas.SaveAs('CP.pdf')



