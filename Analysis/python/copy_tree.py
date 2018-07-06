#!/usr/bin/env python

import ROOT
import argparse

parser = argparse.ArgumentParser(description='Copy selected events.')
parser.add_argument('--input', required=True, type=str, help="Input tuples")
parser.add_argument('--output', required=True, type=str, help="Output directory")
parser.add_argument('--filelist', required=True, type=str, help="Txt file with input tuple list")
parser.add_argument('--match', required=True, type=str, help="Match requirement: e, mu, tau, jet")
args = parser.parse_args()

#ROOT.gROOT.ProcessLine('ROOT::EnableImplicitMT(6)')

with open(args.filelist, 'r') as f_list:
    file_list = [ f.strip() for f in f_list if len(f.strip()) != 0 ]

chain = ROOT.TChain("taus")
for f in file_list:
    f_name = '{}/{}'.format(args.input, f)
    print('Adding "{}" to the chain.'.format(f_name))
    chain.Add(f_name)

match_list = args.match.split(',')
for match in match_list:
    selection = 'pt > 20 && abs(eta) < 2.3'
    if match == 'e':
        selection += ' && (gen_match==1 || gen_match == 3)'
    elif match == 'mu':
        selection += ' && (gen_match==2 || gen_match == 4)'
    elif match == 'tau':
        selection += ' && gen_match == 5'
    elif match == 'jet':
        selection += ' && gen_match == 6'
    else:
        raise RuntimeError('Invalid match requirement = "{}".'.format(match))

    print("Copying taus ({})...".format(match))
    f_out = ROOT.TFile("{}/taus_{}.root".format(args.output, match), "RECREATE")
    f_out.SetCompressionAlgorithm(4) # LZ4
    f_out.SetCompressionLevel(5)
    taus_out = chain.CopyTree(selection)
    f_out.WriteTObject(taus_out, "taus", "Overwrite")
    f_out.Close()
