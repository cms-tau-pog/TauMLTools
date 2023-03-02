## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re
import sys
import shutil

from Analysis.law.framework import Task, HTCondorWorkflow
import luigi

class ShuffleMergeSpectral(Task, HTCondorWorkflow, law.LocalWorkflow):
  ## '_' will be converted to '-' for the shell command invocation
  cfg               = luigi.Parameter(description = 'configuration file with the list of input sources')
  input_path        = luigi.Parameter(description = 'Input file with the list of files to read. '
                                                    'The --prefix argument will be placed in front of --input.')
  output_path       = luigi.Parameter(description = 'output directory')
  pt_bins           = luigi.Parameter(description = 'pt bins')
  eta_bins          = luigi.Parameter(description = 'eta bins')
  input_spec        = luigi.Parameter(description = '')
  tau_ratio         = luigi.Parameter(description = '')
  n_jobs            = luigi.IntParameter(description = 'number of HTCondor jobs to run')
  ## optional arguments (default will be an empty string: don't override ShuffleMergeSpectral.cxx default values)
  prefix            = luigi.Parameter(description = 'Prefix to place before the input file path read from --input.'
                                                    'It can include a remote server to use with xrootd.', default = "")
  mode              = luigi.Parameter(default = '', description = 'merging mode: MergeAll or MergePerEntry')
  compression_algo  = luigi.Parameter(default = '', description = 'ZLIB, LZMA, LZ4')
  compression_level = luigi.Parameter(default = '', description = 'compression level of output file')
  disabled_branches = luigi.Parameter(default = '', description = 'disabled-branches list of branches to disabled in the input tuples')
  parity            = luigi.Parameter(default = '', description = 'take only even:0, take only odd:1, take all entries:3')
  max_entries       = luigi.Parameter(default = '', description = 'maximal number of entries in output train+test tuples')
  n_threads         = luigi.Parameter(default = '', description = 'number of threads')
  lastbin_disbalance= luigi.Parameter(default = '', description = 'maximal acceptable disbalance between low pt and high pt region')
  lastbin_takeall   = luigi.Parameter(default = '', description = 'to take all events from the last bin up to acceptable disbalance')
  seed              = luigi.Parameter(default = '', description = 'random seed to initialize the generator used for sampling')
  enable_emptybin   = luigi.Parameter(default = '', description = 'enable empty pt-eta bins in the spectrum')
  refill_spectrum   = luigi.Parameter(default = '', description = 'to recalculated spectrums of the input data on flight')
  overflow_job      = luigi.Parameter(default = '', description = 'to consider remaining taus in last job')

  def create_branch_map(self):
    self.output_dir = '/'.join([self.output_path, 'tmp'])
    ## create the .root file output directory
    if not os.path.exists(os.path.abspath(self.output_dir)):
      os.makedirs(os.path.abspath(self.output_dir))
    ## this directory will store the json has dictionaries
    if not os.path.exists(os.path.abspath('/'.join([self.output_dir, '..', 'hashes']))):
      os.makedirs(os.path.abspath('/'.join([self.output_dir, '..', 'hashes'])))

    return {i: i for i in range(self.n_jobs)}

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def move(self, src, dest):
    if os.path.exists(dest):
      if os.path.isdir(dest): shutil.rmtree(dest)
      else: os.remove(dest)
    shutil.move(src, dest)

  def run(self):
    self.output_dir = '/'.join([self.output_path, 'tmp'])
    file_name   = '_'.join(['ShuffleMergeSpectral', str(self.branch)]) + '.root'
    output_name = '/'.join([self.output_dir, file_name])

    if not (self.mode == 'MergeAll' or self.mode == ''):
      raise Exception('Only --mode MergeAll is supported by the law tool')

    quote = lambda x: str('\"{}\"'.format(str(x)))
    command = ['ShuffleMergeSpectral',
      '--cfg'               , str(self.cfg)             ,
      '--input'             , str(self.input_path)      ,
      '--output'            , output_name               ,
      '--pt-bins'           , quote(str(self.pt_bins))  ,
      '--eta-bins'          , quote(str(self.eta_bins)) ,
      '--input-spec'        , str(self.input_spec)      ,
      '--n-jobs'            , str(self.n_jobs)          ,
      '--job-idx'           , str(self.branch_data)     ,
      '--tau-ratio'         , quote(str(self.tau_ratio))]
    optional_arguments = [
      ['--mode'              , str(self.mode)                    ],
      ['--seed'              , str(self.seed)                    ],
      ['--prefix'            , str(self.prefix)                  ],
      ['--n-threads'         , str(self.n_threads)               ],
      ['--disabled-branches' , quote(str(self.disabled_branches))],
      ['--lastbin-disbalance', str(self.lastbin_disbalance)      ],
      ['--lastbin-takeall'   , str(self.lastbin_takeall)         ],
      ['--compression-algo'  , str(self.compression_algo)        ],
      ['--compression-level' , str(self.compression_level)       ],
      ['--parity'            , str(self.parity)                  ],
      ['--max-entries'       , str(self.max_entries)             ],
      ['--enable-emptybin'   , str(self.enable_emptybin)         ],
      ['--refill-spectrum'   , str(self.refill_spectrum)         ],
      ['--overflow-job'      , str(self.overflow_job)            ],
    ]
    for arg in optional_arguments:
      if len(arg[1]) > 0:
        command += arg

    command = ' '.join(command)
    print ('>> {}'.format(command))
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()

    sys.stdout.write(str(stdout) + '\n')
    sys.stderr.write(str(stderr) + '\n')

    retcode = proc.returncode

    if retcode != 0:
      raise Exception('job {} return code is {}'.format(self.branch, retcode))
    else:
      self.move(os.path.abspath(output_name), os.path.abspath('/'.join([self.output_dir, '..', file_name])))
      self.move(os.path.abspath('./out'    ), os.path.abspath('/'.join([self.output_dir, '..', 'hashes', 'out_{}'.format(self.branch)])))
      print('Output file and hash tables moved to {}\n'.format(os.path.abspath('/'.join([self.output_dir, '..']))))
      taskout = self.output()
      taskout.dump('Task ended with code %s\n' %retcode)
