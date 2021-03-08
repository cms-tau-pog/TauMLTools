import argparse

parser = argparse.ArgumentParser(description='Create uniform (pt, eta) weights.')
parser.add_argument('--input', required=True, type=str, help="Input tuples")
parser.add_argument('--output', required=True, type=str, help="Output file")
args = parser.parse_args()

from common import *
import sf_calc
from WeightManager import WeightManager

Y_raw = ReadBranchesTo2DArray(args.input, 'taus', truth_branches, int)
Y = VectorizeGenMatch(Y_raw, int)

weightManager = WeightManager(args.output, calc_weights=True, full_file_name=args.input, Y=Y)
