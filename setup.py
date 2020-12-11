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
#long_description='See documentation at https://aurora-fusion.readthedocs.io'

wrapper = Extension(name='aurora._aurora', 
                    sources=['aurora/main.f90',
                             'aurora/grids.f90',
                             'aurora/impden.f90',
                             'aurora/math.f90'])

aurora_dir = os.path.dirname(os.path.abspath(__file__))
install_requires = open('requirements.txt').read().split('\n')

setup(
    name=package_name,
    version='1.3.2',
    description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fsciortino/Aurora',
    author='F. Sciortino',
    author_email='sciortino@psfc.mit.edu',
    packages=['aurora'], #setuptools.find_packages(),
    #package_dir = {'examples': 'examples'},
    #package_data={'examples': ['examples/*']},
    data_files = [('examples', ['examples/basic.py',
                                'examples/frac_abundances.py',
                                'examples/example.gfile',
                                'examples/example.input.gacode'])],
    setup_requires=["numpy"],
    install_requires=install_requires,
    ext_modules=[wrapper],
    classifiers=['Programming Language :: Python :: 3',
                 'Operating System :: OS Independent',
    ],
)
