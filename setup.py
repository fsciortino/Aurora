from numpy.distutils.core import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()


wrapper = Extension(name='aurora', 
                    sources=['aurora/main.f90',
                             'aurora/grids.f90',
                             'aurora/impden.f90',
                             'aurora/math.f90'])

setup(name='aurora',
      version='1.3.1',
      description=long_description,
      url='https://github.com/fsciortino/aurora',
      author='F. Sciortino',
      author_email='sciortino@psfc.mit.edu',
      license='MIT',
      packages=['aurora'],
      install_requires=['numpy','scipy','matplotlib','omfit_eqdsk'],
      ext_modules=[wrapper]
  )

