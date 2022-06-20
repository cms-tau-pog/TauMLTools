import json
import os
import re
import subprocess

pypi_to_conda_matches = {
    "importlib-resources": "importlib_resources",
    "PyYAML": "pyyaml",
    "typing-extensions": "typing_extensions",
}

ignore_list = [ '__glibc', 'antlr4-python3-runtime' ]

packages = {}
class Package:
    def __init__(self, pkg_line):
        pkg_desc = [ s for s in pkg_line.split(' ') if len(s) > 0 ]
        if len(pkg_desc) < 3 or len(pkg_desc) > 5:
            raise RuntimeError(f"Wrong pkg desc = '{pkg_line}'")
        self.name = pkg_desc[0]
        self.version = pkg_desc[1]
        self.build = pkg_desc[2]
        self.channel = ' '.join(pkg_desc[3:])
        self.line = pkg_line
        self.deps = None
        if self.channel == "pypi":
            self.pypi_name = self.name
            if self.name in pypi_to_conda_matches:
                self.name = pypi_to_conda_matches[self.pypi_name]

    def dependsOn(self, other_name, prev_pkgs = []):
        if self.name not in prev_pkgs and self.deps is not None:
            for dep in self.deps:
                if dep == other_name or packages[dep].dependsOn(other_name, prev_pkgs + [self.name]):
                    return True
        return False

def run_cmd(cmd, prefix=''):
    print(prefix + f'>> {cmd}')
    result = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Error while running '{cmd}'")
    return result.stdout.decode('utf-8')

for line in run_cmd('conda list').split('\n'):
    if len(line) == 0 or line[0] == '#': continue
    pkg = Package(line)
    packages[pkg.name] = pkg

requested_packages = []
requested_file = 'requested.txt'
if os.path.exists(requested_file):
    print(f"Reading '{requested_file}'...")
    with open(requested_file, 'r') as f:
        for line in f.readlines():
            requested_packages.append(line.strip())

required_packages = set()

deps_file = 'deps.json'
if os.path.exists(deps_file):
    print(f"Reading '{deps_file}'...")
    with open(deps_file, 'r') as f:
        dep_data = json.load(f)
        for pkg_name, pkg_deps in dep_data.items():
            if pkg_name in packages:
                packages[pkg_name].deps = pkg_deps

def FillRequired(pkg_name, level=0):
    prefix = '  ' * level if level > 0 else ''
    print(prefix + f'Filling requirements for {pkg_name}')
    prefix += '  '
    if pkg_name not in packages:
        raise RuntimeError(f"{pkg_name} not found.")
    pkg = packages[pkg_name]
    if pkg.name in required_packages: return
    required_packages.add(pkg.name)
    if pkg.deps is None:
        print(prefix + f'package={pkg.name} channel={pkg.channel}')
        if pkg.channel == "pypi":
            pkg_deps_str = run_cmd(f'pipdeptree -p {pkg.pypi_name}', prefix)
            pkg.deps = []
            for dep_str in pkg_deps_str.split('\n'):
                result = re.search('^[ ]*- ([^ ]+).*$', dep_str)
                if result:
                    dep_pkg_name = result.group(1)
                    if dep_pkg_name in pypi_to_conda_matches:
                        dep_pkg_name = pypi_to_conda_matches[dep_pkg_name]
                    pkg.deps.append(dep_pkg_name)
        else:
            pkg_deps_str = run_cmd(f'conda-tree depends {pkg.name}', prefix)
            pkg.deps = eval(pkg_deps_str)
    pkg.deps = [ d for d in pkg.deps if d not in ignore_list ]
    for pkg_dep in pkg.deps:
        FillRequired(pkg_dep, level+1)

for pkg_name in requested_packages:
    FillRequired(pkg_name)

with open(deps_file, 'w') as f:
    dep_data = {}
    for pkg_name in sorted(packages):
        pkg = packages[pkg_name]
        if pkg.deps is not None:
            dep_data[pkg_name] = pkg.deps
    json.dump(dep_data, f)

nonrequired_packages = []
for pkg_name in sorted(packages):
    if pkg_name not in required_packages and pkg_name not in ignore_list:
        nonrequired_packages.append(pkg_name)

if len(nonrequired_packages) > 0:
    print("Nonrequired packages:")
    for pkg_name in nonrequired_packages:
        print(packages[pkg_name].line)
    print("Command line to remove nonrequired packages:")
    print('conda remove ' + ' '.join(nonrequired_packages))
else:
    print("No nonrequired packages is found.")

print("Checking dependencies between requested packages...")
dep_found = False
for pkg_name in sorted(requested_packages):
    pkg = packages[pkg_name]
    for other_pkg_name in sorted(requested_packages):
        if pkg_name == other_pkg_name: continue
        if pkg.dependsOn(other_pkg_name):
            print(f'{pkg_name} depends on {other_pkg_name}')
            dep_found = True
if not dep_found:
    print("No dependencies between requested packages is found.")
