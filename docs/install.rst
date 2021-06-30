Installation
============

Installing from source
----------------------

We recommend installing from the latest version of the code, obtained by git-cloning the repository at

    https://github.com/fsciortino/aurora
    
After doing this, you can run::

  python setup.py install

which should do the magic.

Some users may want to have greater control over which compiler is being used for the installation; this can be most easily done by modifying the provided Makefile directly. After changing its top configuration lines, users can do::

  make clean; make


Installing via PyPI or Anaconda
-------------------------------

The latest release is now available via PyPI using::

  pip install aurorafusion

.. image:: https://badge.fury.io/py/aurorafusion.svg
    :target: https://badge.fury.io/py/aurorafusion

	     
Installing via conda is also possible using

    conda install -c conda-forge aurorafusion

.. image:: https://anaconda.org/conda-forge/aurorafusion/badges/version.svg   
    :target: https://anaconda.org/conda-forge/aurorafusion



Running with Julia
------------------

Aurora simulations can also be done using a Python-Julia interface; this makes iterative runs even faster!

Assuming that you have Julia already installed on your device, you will want to build a `sysimage` for the Aurora Julia source code. This is useful because whenever you will open a Python session the first run of Aurora using :py:meth:`~aurora.core.run_aurora` will need to pre-compile the Julia source code, which may take a couple of seconds. To create the `sysimage`, you can do::

  make clean_julia; make julia

This may take a couple of minutes, but it only has to be done once. 

Once the `sysimage` has been created, Python can directly make use of it and enjoy even greater speed. Note that this is only recommended for "iterative" operation, i.e. when many Aurora simulations are run within the same Python session, since the first run will take much longer than usual. All the following simulations will be faster.

Of course, interfaces to run Aurora completely in Julia are under-development (@ajcav). Interested parties should get in touch! 


It may be surprising that Julia can beat good-old Fortran at what it is normally best (speed). Well, we all get used to it after some time :)


What's next?
------------

After installing, see the :ref:`Tutorial` section for guidance on how to get started.

