import sys
import subprocess
import glob
import shutil
import os
import platform

matrix_os, = sys.argv[1:]

so_name, = glob.glob(f'libtvbk-{matrix_os}/libtvbk.so')
print(f'[{__file__}] found artifact {so_name}')

whl_name, = glob.glob('tvbk-kernels-wheel/tvb_kernels*.whl')
print(f'[{__file__}] installing {whl_name}')
os.system(f'{sys.executable} -m pip install {whl_name}')

import site
tvbk_name = None
for pkg in site.getsitepackages():
    fs = glob.glob(os.path.join(pkg, 'tvbk.py'))
    if fs:
        tvbk_name, = fs
        print(f'[{__file__}] found {tvbk_name}')
        break

so_dst = os.path.join(os.path.dirname(tvbk_name), 'libtvbk.so')
shutil.copy(so_name, so_dst)
    
