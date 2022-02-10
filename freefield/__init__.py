import pathlib
import sys
__version__ = '1.1.0'

sys.path.append('..\\')
DIR = pathlib.Path(__file__).parent.resolve()

from freefield.processors import Processors
from freefield.cameras import Cameras
from freefield.freefield import *
