# python merge_feature_scaling_jobs.py --cfg /path/to/yaml --json /path/to/job*/json/file.json --output /path/to/new/json/file.json
# if --step is given, this will produce /path/to/new/json/file{N}.json files, where N is each step of logging
# if --var-types is given, it will only scan the selected var type
import json
import yaml
import glob
from collections import OrderedDict

def merge_jobs(jobs, output_path):  
  variable_types = [k for k in features_dict.keys() if k in args.var_types or args.var_types is None]
  odict = OrderedDict()  # output dictionary
  # loop inside the variable types (TauFlat, GridGlobal, etc.)
  for vt in variable_types:
    odict[vt] = OrderedDict()
    # load the yaml cfg corresponding to the variable type
    var_cfg = features_dict[vt]
    # loop inside the features of the selected variable type   
    for var, desc in var_dict.iteritems():
      odict[vt][var] = OrderedDict()
      # get the json subjeys correspondiing to the selected feature
      dicts = [j[vt][var] for j in jobs]
      # loop inside the dictionary keys (cone type)
      for key in dicts[0].keys():
        odict[vt][var][key] = OrderedDict()

        lmin = dicts[0][key]['lim_min']
        lmax = dicts[0][key]['lim_max']

        # check the consistency of the input json files (jobs)
        thisstep = "{}/{}/{}".format(vt,var,key)
        assert all(key in d.keys() for d in dicts), "Not all the job json files have the same structure for "+thisstep
        assert all(d[key]['lim_min'] == lmin for d in dicts), "lim_min parameter was found to be different between jobs for "+thisstep
        assert all(d[key]['lim_max'] == lmax for d in dicts), "lim_max parameter was found to be different between jobs for "+thisstep

        # load mean, mean of the square, std and number of entries for each job
        means   = [d[key]['mean']         for d in dicts]
        sqmeans = [d[key]['square_mean']  for d in dicts]
        stds    = [d[key]['std']          for d in dicts]
        nums    = [d[key]['entries']      for d in dicts]
        # merge the information above into a single structure
        num     = sum(nums)
        sqmean  = sum(m*n for m,n in zip(sqmeans, nums)) / num # FIXME: is this correct? 
        odict[vt][var]['mean'] = sum(m*n for m,n in zip(means  , nums)) / num # FIXME: is this correct?
        odict[vt][var]['std']  = sqmean-mean**2 # FIXME: is this correct?
        odict[vt][var]['lim_min'] = lmin
        odict[vt][var]['lim_max'] = lmax
  
  with open(output_path, 'w') as ojson:
    json.dump(ojson, odict, indent = 4)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg'      , required = True, help = 'yaml cfg file used to produce the feature scaling parameters')
  parser.add_argument('--output'   , required = True, help = 'path and name of the output file')
  parser.add_argument('--json'     , required = True, help = 'path to json files storing the job results. Accepts glob patterns')
  parser.add_argument('--step'     , default  = None, help = 'step for logging the convergence of the computation. None = skip')
  parser.add_argument('--var-types', default  = None, help = 'variable types to read, space separated', nargs = '+') 
  args = parser.parse_args()

  # load the job json files into dictionaries
  alljobs = [json.load(j) for j in glob.glob(args.json)]

  with open(args.cfg) as f:
    scaling_dict = yaml.load(f, Loader=yaml.FullLoader)

  features_cfg = scalinig_dict['Features_all']

  if args.step is not None:
    for ii, st in enumerate(range(0, len(alljobs), args.step)):
      merge_jobs(jobs = alljobs[:st], output_path = args.output_name.replace('.json', '_{}.json'.format(ii)))
  else:
    merge_jobs(jobs = alljobs[:st], output_path = args.output_name)
  
  print('All done.  \n\
  Report:           \n\
  \tcfg: {C}        \n\
  \tinput: {I}      \n\
  \toutput: {O}     \n\
  \tlog step: {S}   \n\
  \tvar. types: {T}'''.format(
    C=args.cfg,
    I=args.json,
    O=args.output,
    S=args.step if args.step is not None else 'skipped', 
    T=args.var_types if args.var_types is not None else 'all'
  )