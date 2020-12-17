# Freefield #

The code in this package was written to conduct experiments using a psychoacoustics setup which consists of 48 loudspeakers and 4 cameras in a hemi-anaechoic room. The loudspeakers are driven by two digital signal processors
from Tucker-Davis Technologies (TDT). While large parts of this package are tailored to one specific experimental
setup, some elements (e.g. handling TDT device, head pose estimation) might have broader applicability.

## Installation ##

To use the functionalities of the processors you have to download and install the drivers from the
[TDT Hompage](https://www.tdt.com/support/downloads/ "TDT Downloads") (install TDT Drivers/RPvdsEx
as well as ActiveX Controls). Note that these drivers can only be installed on a Windows machine.

You will also need Python version 3.6 or greater (If you don't have Python installed, take a look
at the installation guide for the [Anaconda](https://docs.anaconda.com/anaconda/install/ "Install Anaconda")) distribution.

If you have pip and the git installed (assuming you are using Anaconda you can do so by
typing `conda install pip git`), you can install freefield from GitHub: \
`pip install git+https://github.com/OleBialas/freefield.git`

Another package that needs to be installed from github is slab:
`pip install git+https://github.com/DrMarc/soundlab.git`
Slab is used to manipulate sounds and run experiments, if you are studying
psychoacoustics you might want to
[check it out](https://soundlab.readthedocs.io/en/latest/?badge=latest "Slab Documentation")!

The communication with the processors relies on the pywin32 package. Since installing it with pip can result
in a faulty version, using conda is preferred (if you want to use features of freefield on a different OS than
Windows, leave this step out):
`conda install pywin32`

All other python dependencies can be installed using pip:
`pip install tensorflow opencv-python numpy setuptools pandas matplotlib pillow scipy`

To use cameras from the manufacturer FLIR systems, you have to install their Python API (Python version >3.8 is not supported). Go to the [download page](https://meta.box.lenovo.com/v/link/view/a1995795ffba47dbbe45771477319cc3 "Spinnaker Download")! and select the correct file for your OS and Python version. For example, if you are using
a 64-Bit Windows and Python 3.8 download spinnaker_python-2.2.0.48-cp38-cp38-win_amd64.zip.
Unpack the .zip file and select the folder. There should be a file inside that ends with .whl - install it using pip:
`pip install spinnaker_python-2.2.0.48-cp38-cp38-win_amd64.whl`

To check if everything was installed correctly, you can run a sound localization test

## Example: Sound Localization Test ##
