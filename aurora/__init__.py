name = "aurora"
__version__ = "2.1.2"

import numpy as np, os

from .core import *
from .atomic import *
from .adas_files import *
from .amdata import *

from .source_utils import *
from .default_nml import *
from .grids_utils import *
from .coords import *
from .grids_utils import *
from .radiation import *
from .amdata import *

from .plot_tools import *
from .animate import *

from .janev_smith_rates import *
from .nbi_neutrals import *
from .neutrals import *

from .synth_diags import *

from .solps import *
from .kn1d import *
from .oedge import *

aurora_dir = os.path.dirname(os.path.abspath(__file__))
