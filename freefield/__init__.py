import pathlib
import os
import sys

__version__ = '0.1'

sys.path.append('..\\')
DATADIR = str(pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')) + os.sep
