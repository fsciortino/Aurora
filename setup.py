from numpy.distutils.core import setup, Extension
import os

with open("README.md", "r") as fh:
    long_description = fh.read()


wrapper = Extension(name='aurora-fusion', 
                    sources=['aurora/main.f90',
                             'aurora/grids.f90',
                             'aurora/impden.f90',
                             'aurora/math.f90'])

# load version number from .version file
aurora_dir = os.path.dirname(os.path.abspath(__file__))
with open(aurora_dir+'/.version') as vfile:
    version = vfile.read().strip()

setup(name='aurora-fusion',
      version=version,
      description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/fsciortino/Aurora',
      author='F. Sciortino',
      author_email='sciortino@psfc.mit.edu',
      packages=['aurora'],
      requires=['numpy','scipy','matplotlib','xarray',
                'omfit_commonclasses','omfit_eqdsk','omfit_gapy'],
      ext_modules=[wrapper],
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ],
  )

