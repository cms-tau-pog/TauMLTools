import sys

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Patch exported HLT configurations for Tau HLT production.')
  parser.add_argument('--input', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--patch', required=True, type=str)
  parser.add_argument('--last-input-line', required=False, type=str,
                      default='process.schedule.append( process.DQMOutput )')
  parser.add_argument('--type', required=True, type=str, help='data or mc')
  args = parser.parse_args()

  if args.type not in ['data', 'mc']:
    print('Invalid type: {}'.format(args.type))
    sys.exit(1)
  is_data = args.type == 'data'

  input_lines = []
  last_line_found = False
  with open(args.input, 'r') as f:
    for line in f.readlines():
      input_lines.append(line)
      if line.strip() == args.last_input_line:
        last_line_found = True
        break
  if not last_line_found:
    print('Last line not found in the input file.')
    sys.exit(1)
  with open(args.patch, 'r') as f:
    patch_lines = f.readlines()
  with open(args.output, 'w') as f:
    for line in input_lines:
      f.write(line)
    for line in patch_lines:
      line = line.replace('IS_DATA', str(is_data))
      f.write(line)