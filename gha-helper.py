import sys
import subprocess
import glob
import shutil
import os
import platform
import importlib.util

matrix_os, = sys.argv[1:]

def log(msg):
    print(f'[{__file__}] {msg}', flush=True)

log(f'have {matrix_os} with arch {platform.processor()}')

so_name, = glob.glob(f'libtvbk-{matrix_os}/libtvbk.*')
base_so_name = os.path.basename(so_name)
log(f'found artifact {so_name}')

whl_name, = glob.glob('tvbk-kernels-wheel/tvb_kernels*.whl')
log(f'installing {whl_name}')
os.system(f'{sys.executable} -m pip install {whl_name}')

tvbk_name = importlib.util.find_spec('tvbk').origin
log(f'found {tvbk_name}: juxtaposing {base_so_name}')
so_dst = os.path.join(os.path.dirname(tvbk_name), base_so_name)
shutil.copy(so_name, so_dst)
    
