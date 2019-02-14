#!/usr/bin/env python

import os
import sys
import fnmatch
import uproot

if len(sys.argv) != 2:
    print("Usage: tuples_path")
    sys.exit(1)

input = sys.argv[1]
if not os.path.isdir(input):
    raise RuntimeError("Input directory '{}' not found".format(input))

for root_dir, dir_names, file_names in os.walk(input):
    for file_name in fnmatch.filter(file_names, '*.root'):
        full_file_name = os.path.join(root_dir, file_name)
        rel_file_name = full_file_name[len(input) + 1:]
        with uproot.open(full_file_name) as file:
            keys = [ key[:key.rindex(b';')] for key in file.keys() ]
            if 'taus' in keys:
                tree = file['taus']
                print("{} {}".format(rel_file_name, tree.numentries))
