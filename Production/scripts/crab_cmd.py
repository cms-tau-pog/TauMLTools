#!/usr/bin/env python
# Execute crab command for multiple tasks.

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description='Execute crab command for multiple tasks.',
                  formatter_class = lambda prog: argparse.HelpFormatter(prog,width=90))
parser.add_argument('--workArea', required=True, type=str, help="Working area")
parser.add_argument('--cmd', required=True, type=str, help="CRAB command")
parser.add_argument('cmd_args', type=str, nargs='*', help="Arguments for the CRAB command (if any)")
args = parser.parse_args()

if args.cmd == 'submit':
    print("ERROR: Please, use crab_submit.py to run the submit command.")
    sys.exit(1)
cmd_args_str = ' '.join(args.cmd_args)

def sh_call(cmd):
    sep = '-' * (len(cmd) + 3)
    print('{}\n>> {}'.format(sep, cmd))
    result = subprocess.call([cmd], shell=True)
    if result != 0:
        print('ERROR: failed to run "{}"'.format(cmd))
        sys.exit(1)

for dir in os.listdir(args.workArea):
    task_dir = os.path.join(args.workArea, dir)
    if not os.path.isdir(task_dir): continue

    cmd_line = 'crab {} -d {} {}'.format(args.cmd, task_dir, cmd_args_str)
    sh_call(cmd_line)
