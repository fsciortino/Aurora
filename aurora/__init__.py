name='aurora'

""":py:mod:`aurora` 
"""

#with open('./VERSION') as version_file:
#    version = version_file.read().strip()

__version__ = '1.2.1'


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from ._aurora import run,get_radial_grid,time_steps
from .atomic import *
#from .utils import *
from .core import *
from .source_utils import *
from .default_nml import *
from .grids_utils import *
from .particle_conserv import *
from .plot_tools import *
from .coords import *
from .radiation import *
from .animate import *
