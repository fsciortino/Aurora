'''
Setup for Aurora package. Basic call (install in editable mode)
pip install -e .

To install with a different Fortran compiler, use e.g.
python3 setup.py build --fcompiler=gnu95
or
python3 setup.py build --fcompiler=intelem 

It should be possible to pass any f2py flags via the command line, e.g. using
python3 setup.py build --fcompiler=intelem --opt="-fast"

'''

import setuptools
import os, sys, subprocess
from numpy.distutils.core import setup, Extension

package_name='aurorafusion'

with open('README.md', 'r') as fh:
    long_description = fh.read()

wrapper = Extension(name='aurora._aurora', 
                    sources=['aurora/main.f90',
                             'aurora/grids.f90',
                             'aurora/impden.f90',
                             'aurora/math.f90'])

# use local makefile and avoid numpy's Extension class...
#cmd = 'make clean; make aurora'
#result = subprocess.call(cmd, shell=True) 


aurora_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(aurora_dir, 'aurora', 'version')) as vfile:
    version = vfile.read().strip()

install_requires = open('requirements.txt').read().split('\n')
    
setup(name=package_name,
      version=version,
      description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/fsciortino/Aurora',
      author='F. Sciortino',
      author_email='sciortino@psfc.mit.edu',
      packages=['aurora'], #setuptools.find_packages(),
      setup_requires=["numpy"],
      install_requires=install_requires,
      include_package_data=True,
      package_data={'':['aurora/version']},
      ext_modules=[wrapper],
      classifiers=['Programming Language :: Python :: 3',
                   'Operating System :: OS Independent',
                   ],
      )

# move shared-object library to ./aurora
#filename = [filename for filename in os.listdir('.') if filename.startswith('_aurora')]
#print(filename)
#os.rename(filename, './aurora/'+filename)
