from distutils.core import setup

setup(
    name='freefield_toolbox',
    version='0.1dev',
    packages=['freefield'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    include_package_data=True, requires=['numpy', 'dlib', 'opencv', 'imutils', 'scikit-learn', 'matplotlib']
)
