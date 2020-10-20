import pathlib
import sys
__version__ = '0.1'

sys.path.append('..\\')
DATADIR = pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')

from freefield.devices import Devices
