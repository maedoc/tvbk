import os

with open('pyproject.toml', 'r') as fd:
    project = fd.readlines()
newlines = []
maj, min = -1, -1
for line in project:
    if line.startswith('version'):
        _, v = line.split('=')
        v = eval(v)
        maj, min = [int(_) for _ in v.split('.')]
        next = f'version = "{maj}.{min+1}"\n'
        newlines.append(next)
    else:
        newlines.append(line)
with open('pyproject.toml', 'w') as fd:
    fd.write(''.join(newlines))
assert min >= 0

os.system('git commit -m "bump version"')
os.system(f'git tag v{maj}.{min}')
os.system(f'git push origin master')
os.system(f'git push v{maj}.{min}')
