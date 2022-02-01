# python merge_feature_scaling_jobs.py --output mydir --input "/path/to/job*/json/file.json"
# if --step is given, this will produce /path/to/new/json/scaling_params_N.json files, where N is each step of logging (cumulative)
# if --var-types is given, it will only scan the selected var type
import json
import yaml
import glob
import math
import os, sys
from collections import OrderedDict

def check(val):
  return not (math.isnan(val) or math.isinf(val))

def merge_jobs(jobs, output_path):
  KERROR = "Input json files have different keys at level "
  odict = OrderedDict()
  variable_types = jobs[0].keys()

  thisstep = '/'

  assert all(j.keys() == variable_types for j in jobs), KERROR+thisstep

  for vt in variable_types:
    thisstep  = '/'+vt+'/'
    odict[vt] = OrderedDict()
    variables = jobs[0][vt].keys()

    assert all(j[vt].keys() == variables for j in jobs), KERROR+thisstep

    for var in variables:
      thisstep = '/'+vt+'/'+var+'/'
      odict[vt][var] = OrderedDict()
      cones = jobs[0][vt][var].keys()

      assert all(j[vt][var].keys() == cones for j in jobs), KERROR+thisstep

      for ct in cones:
        thisstep = '/'+vt+'/'+var+'/'+ct+'/'
        odict[vt][var][ct] = OrderedDict()
        stats = jobs[0][vt][var][ct].keys()

        assert all(j[vt][var][ct].keys() == stats for j in jobs), KERROR+thisstep

        lmin = jobs[0][vt][var][ct]['lim_min']
        lmax = jobs[0][vt][var][ct]['lim_max']

        assert all(j[vt][var][ct]['lim_min'] == lmin for j in jobs), "lim_min parameter was found to be different between jobs for "+thisstep
        assert all(j[vt][var][ct]['lim_max'] == lmax for j in jobs), "lim_max parameter was found to be different between jobs for "+thisstep

        # load mean, mean of the square, std and number of entries for each job
        assert all('mean'   in j[vt][var][ct].keys() for j in jobs), "'mean' key not found for some of the jobs at "+thisstep
        assert all('sqmean' in j[vt][var][ct].keys() for j in jobs), "'sqmean' key not found for some of the jobs at "+thisstep
        assert all('std'    in j[vt][var][ct].keys() for j in jobs), "'std' key not found for some of the jobs at "+thisstep
        assert all('num'    in j[vt][var][ct].keys() for j in jobs), "'num' key not found for some of the jobs at "+thisstep

        means   = [j[vt][var][ct]['mean']   for j in jobs]
        sqmeans = [j[vt][var][ct]['sqmean'] for j in jobs]
        stds    = [j[vt][var][ct]['std']    for j in jobs]
        nums    = [j[vt][var][ct]['num']    for j in jobs]

        assert all(check(v) for v in means  ), "'mean' value found 'nan' or 'inf' for some of the jobs at "+thisstep
        assert all(check(v) for v in stds   ), "'stds' value found 'nan' or 'inf' for some of the jobs at "+thisstep

        # merge the information above into a single structure
        if any(x is None for x in sqmeans) or any(x is None for x in nums):
          assert all(x is None for x in sqmeans), "All sq. mean should be None but they are not at "+thisstep
          assert all(x is None for x in nums)   , "All n.events should be None but they are not at "+thisstep

          odict[vt][var][ct]['mean'] = jobs[0][vt][var][ct]['mean']
          odict[vt][var][ct]['std']  = jobs[0][vt][var][ct]['std' ]

          assert all(j[vt][var][ct]['mean'] == odict[vt][var][ct]['mean'] for j in jobs), "All means should be equal but they are not at "+thisstep
          assert all(j[vt][var][ct]['std' ] == odict[vt][var][ct]['std' ] for j in jobs), "All std's should be equal but they are not at "+thisstep

        else:
          assert all(check(v) for v in sqmeans), "'sqmeans' value found 'nan' or 'inf' for some of the jobs at "+thisstep
          assert all(check(v) for v in nums   ), "'nums' value found 'nan' or 'inf' for some of the jobs at "+thisstep

          num     = sum(nums)
          mean    = sum(m*n for m,n in zip(means  , nums)) / num
          sqmean  = sum(m*n for m,n in zip(sqmeans, nums)) / num
          odict[vt][var][ct]['mean'] = float(format(mean, '.4g'))
          odict[vt][var][ct]['std']  = float(format(math.sqrt(sqmean-mean**2), '.4g'))

        odict[vt][var][ct]['lim_min'] = lmin
        odict[vt][var][ct]['lim_max'] = lmax

  with open(output_path, 'w') as ojson:
    json.dump(odict, ojson, indent = 4)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--output'   , required = True, type = str, help = 'path to the output directory')
  parser.add_argument('--input'    , required = True, type = str, help = 'path to json files storing the jobs results. Accepts glob patterns (use quotes)')
  parser.add_argument('--step'     , default  = None, type = int, help = 'step for logging the convergence of the computation. None = skip')
  args = parser.parse_args()

  # load the job json files into dictionaries
  alljobs = [json.load(open(j, 'r')) for j in glob.glob(args.input)]
  if not os.path.exists(args.output):
    os.makedirs(args.output)
  else:
    raise Exception("Directory {} already exists".format(args.output))
  output_path = args.output+'/scaling_params.json'
  if args.step is not None:
    for ii, st in enumerate(range(0, len(alljobs)-args.step, args.step)):
      prog = '\rMergin into step files: {} / {}'.format(ii+1, math.ceil((len(alljobs)-args.step) / args.step))
      sys.stdout.write(prog) ; sys.stdout.flush()
      merge_jobs(jobs = alljobs[:st+args.step], output_path = output_path.replace('.json', '_log_{}.json'.format(ii)))
  print('\nMerging in a single file')
  merge_jobs(jobs = alljobs, output_path = output_path)

  print('\nAll done. Report: \n\
  input: {I}      \n\
  output: {O}     \n\
  log step: {S}   '''.format(
    I=args.input,
    O=args.output,
    S=args.step if args.step is not None else 'skipped', 
  ))