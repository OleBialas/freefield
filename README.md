![Package](https://github.com/OleBialas/freefield/workflows/Python%20package/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/soundlab/badge/?version=latest)](https://free-field.readthedocs.io/en/latest/?badge=latest)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/OleBialas/freefield/graphs/commit-activity)
![PyPI pyversions](https://img.shields.io/badge/python-%3E%3D3.6-blue)
![PyPI license](https://img.shields.io/badge/license-MIT-brightgreen)
[![DOI](https://zenodo.org/badge/195776894.svg)](https://zenodo.org/badge/latestdoi/195776894)


The code in this package was written to conduct experiments using a psychoacoustics setup which consists of 48 loudspeakers and 4 cameras in a hemi-anaechoic room. The loudspeakers are driven by two digital signal processors
from Tucker-Davis Technologies (TDT). While large parts of this package are tailored to one specific experimental
setup, some elements (e.g. handling TDT device, head pose estimation) might have broader applicability.

#Getting Started

The installation consists of two parts - the Python dependencies and the drivers for the hardware in the experimental
setup. The latter is only relevant if you actually use the devices and is not required if you merely want to play
around with the code.

## Python dependencies ##

You will need Python version 3.8 since this is required by tensorflow which is necessary for head pose estimation.
If you are new to Python, take a look  at the installation guide for the [Anaconda](https://docs.anaconda.com/anaconda/install/ "Install Anaconda") distribution.

Once you installed Anaconda, create a new environment with the correct Python version (name it "freefield" for example): \
`conda create --name freefield python=3.8` \
Activate the environment and install pip, which is necessary to install other Python packages: \
`conda activate freefield` \
`conda install pip` \
Now install the remaining python packages: \
`pip install tensorflow==2.3 opencv-python numpy setuptools pandas matplotlib pillow scipy`
Finally, you have to obtain the freefield package as well as another one from github:
`pip install git+https://github.com/OleBialas/slab.git`
`pip install git+https://github.com/OleBialas/freefield.git`

## Hardware drivers ##

To use the functionalities of the processors you have to download and install the drivers from the
[TDT Hompage](https://www.tdt.com/support/downloads/ "TDT Downloads") (install TDT Drivers/RPvdsEx
as well as ActiveX Controls). Note that these drivers can only be installed on a Windows machine.
The communication with the processors relies on the pywin32 package which can be installed using conda: \
`conda install pywin32` \

To use cameras from the manufacturer FLIR systems, you have to install their Python API (Python version >3.8 is not supported). Go to the [download page](https://meta.box.lenovo.com/v/link/view/a1995795ffba47dbbe45771477319cc3 "Spinnaker Download")! and select the correct file for your OS and Python version. For example, if you are using
a 64-Bit Windows and Python 3.8 download spinnaker_python-2.2.0.48-cp38-cp38-win_amd64.zip.
Unpack the .zip file and select the folder. There should be a file inside that ends with .whl - install it using pip:\
`pip install spinnaker_python-2.2.0.48-cp38-cp38-win_amd64.whl`

