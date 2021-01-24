Atomic data
===========

Almost all of the Aurora functionality depends on having access to Atomic Data and Analysis Structure (ADAS) rates. These are needed to determine effective ionization and recombination rates for all charge states, estimate radiated power, soft X-ray contributions, charge exchange components, etc..

Everything that is needed can be obtained from the OPEN-ADAS website:

  https://open.adas.ac.uk/

Aurora attempts to make atomic data usage as simple as possible. The :py:func:`~aurora.adas_files.adas_files_dict` function gives a dictionary of recommended files that users can adopt (but also easily override, if other files are preferable). See the :py:func:`~aurora.atomic.get_file_types` function docstring for a brief description of each relevant file type.

The `adas_data` directory at the base of the Aurora distribution is where ADAS atomic data should be stored, separately for ADF11 (iso-nuclear master files) and ADF15 (photon emissivity coefficients). When running Aurora, the :py:func:`~aurora.adas_files.get_adas_file_loc` function automatically checks whether the requested ADF11 file is available in `adas_data/adf11/` or in a directory that users may specify by setting an environmental variable `AURORA_ADAS_DIR`. If the requested file is not available here either, Aurora attempts to fetch it automatically from the OPEN-ADAS website. Each ADF11 file is stored in `adas_data` after usage, so downloading over the internet is only done if no other option is available.

Atomic data is also used for radiation predictions, both via ADAS ADF11 files (iso-nuclear master files, giving effective coefficients for combined atomic processes) and via ADF15 files (photon emissivity coefficients - PECs - for specific atomic lines):

(a) A number of functions are available in the :py:mod:`~aurora.radiation` module to plot effective radiation terms, e.g. total line radiation for an ion, main ion bremsstrahlung, etc.

(b) The :py:func:`~aurora.radiation.read_adf15` function allows reading and plotting of ADF15, making it easy to evaluate PECs for specific densities and temperatures by using the returned interpolation functions. PEC components due to excitation, recombination and charge exchange can all be easily loaded and plotted. However, Aurora users may also make use of the coupling to `ColRadPy`_ to produce PECs using ADAS ADF04 files and running ColRadPy's collisional-radiative model. This functionality is already available in the :py:func:`~aurora.radiation.get_colradpy_pec_prof` function and will be further developed in the future.

.. _ColRadPy: https://github.com/johnson-c/ColRadPy

See the tutorial in :ref:`Radiation predictions` for more information on these subjects.
