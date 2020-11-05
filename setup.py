from numpy.distutils.core import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()


wrapper = Extension(name='aurora-fusion', 
                    sources=['aurora/main.f90',
                             'aurora/grids.f90',
                             'aurora/impden.f90',
                             'aurora/math.f90'])

setup(name='aurora-fusion',
      version='1.0.1',
      description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/fsciortino/Aurora',
      author='F. Sciortino',
      author_email='sciortino@psfc.mit.edu',
      packages=['aurora'],
      requires=['numpy','scipy','matplotlib','xarray',
                'omfit_commonclasses','omfit_eqdsk','omfit_gapy'],
      ext_modules=[wrapper]
  )

