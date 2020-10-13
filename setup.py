from setuptools import setup, find_packages
import re

with open('README.md') as f:
    readme = f.read()

# extract version
with open('freefield/__init__.py') as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setup(name='freefield',
      version=version,
      description='Toolbox for Experiments on Spatial Hearing.',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/OleBialas/freefield_toolbox.git',
      author='Ole Bialas',
      author_email='bialas@cbs.mpg.de',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['numpy', 'matplotlib', 'slab'],
      packages=find_packages(),
      package_data={'freefield': ['data/*.txt', 'data/*.wav', 'data/rcx/*']},
      include_package_data=True,
      zip_safe=False)
