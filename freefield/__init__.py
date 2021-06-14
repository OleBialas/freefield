import pathlib
import sys
__version__ = '0.1'

sys.path.append('..\\')
DIR = pathlib.Path(__file__).parent.resolve()

from freefield.processors import Processors
from freefield.camera import Cameras
from freefield.freefield import *
