name='aurora'

""":py:mod:`aurora` 
"""

#with open('./VERSION') as version_file:
#    version = version_file.read().strip()

__version__ = '1.2.1'

# make sure that current directory is in sys.path
import sys, os
mod_dir = os.path.dirname( os.path.abspath(__file__) ) 
sys.path.insert(1, mod_dir)
#print(f'Adding {mod_dir} to sys.path')

from .flib import *
from .pylib import *




