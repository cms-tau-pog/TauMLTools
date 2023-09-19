import ROOT
ROOT.gROOT.SetBatch(True)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Dump branches.')
  parser.add_argument('--tree', required=False, default='Events', type=str, help="name of the TTree")
  parser.add_argument('input', nargs=1, type=str, help="input file")
  args = parser.parse_args()

  df = ROOT.RDataFrame(args.tree, args.input[0])
  columns = [ str(c) for c in df.GetColumnNames() ]
  columns = sorted(columns)
  max_len = max([ len(c) for c in columns ])
  for c in columns:
    c_type = str(df.GetColumnType(c))
    print(f'{c:<{max_len+4}}{c_type}')
