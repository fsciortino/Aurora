from numpy.distutils.core import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('VERSION') as version_file:
    version = version_file.read().strip()


wrapper = Extension(name='aurora', 
                    sources=['flib/main.f90',
                             'flib/grids.f90',
                             'flib/impden.f90',
                             'flib/math.f90'])

setup(name='aurora',
      version=version,
      description=long_description,
      url='https://github.com/fsciortino/aurora',
      author='F. Sciortino',
      author_email='sciortino@psfc.mit.edu',
      license='MIT',
      packages=['flib','pylib'],
      #py_modules=['__init__',
      #            'flib/__init__',
      #            'pylib/__init__',
      #            'pylib/coords',
      #            'pylib/default_nml', 
      #            'pylib/grids_utils',
      #            'pylib/interp', 
      #            'pylib/particle_conserv',
      #            'pylib/plot_tools',
      #            'pylib/source_utils',
      #            'pylib/utils',
      #            'pylib/radiation',
      #            'tests/test',
      #            ], 

      ext_modules=[wrapper]
  )
      #entry_points={
      #   'console_scripts': [
      #      'help = pythontools.help:main',
      #  ],
      #}

