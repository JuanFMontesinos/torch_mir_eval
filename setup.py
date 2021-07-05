import re
from setuptools import setup, find_packages

VERSIONFILE = "torch_mir_eval/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='torch_mir_eval',
      version=verstr,
      description='Mir_eval package ported to pytorch ',
      url='https://github.com/JuanFMontesinos/torch_mir_eval',
      author='Juan Montesinos',
      author_email='juanfelipe.montesinos@upf.edu',
      packages=find_packages(),
      install_requires=['torch>=1.9'],
      classifiers=[
          "Programming Language :: Python :: 3", ],
      zip_safe=False)
