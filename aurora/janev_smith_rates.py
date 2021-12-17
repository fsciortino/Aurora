'''
Script collecting rates from Janev & Smith, NF 1993. 
These are useful in aurora to compute total (n-unresolved) charge exchange rates between heavy ions and neutrals. 
'''
# MIT License
#
# Copyright (c) 2021 Francesco Sciortino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import copy


def js_sigma_ioniz_n1_q8(E):
    """Ionization cross section for

    .. math::
    
        O^{8+} + H(1s) --> O^{8+} + H^+ +e^-

    Notes
    ---------
    Section 4.2.4 of Janev & Smith, NF 1993. 
    """
    A1 = 1244.44
    A2 = 249.36
    A3 = 30.892
    A4 = 9.0159e-4
    A5 = 7.7885e-3
    A6 = -0.71309
    A7 = 3.2918e3
    A8 = -2.7541
    return 1e-16 * A1 * ((np.exp(-A2 / E) * np.log(1.0 + A3 * E)) / E + (A4 * np.exp(-A5 * E)) / (E ** A6 + A7 * E ** A8))  # cm^2


def js_sigma_cx_n1_q1(E):
    """Electron capture cross section for

    .. math::

        H^{+} + H(1s) --> H + H^+

    Notes
    ---------
    Section 2.3.1 of Janev & Smith, NF 1993. 
    """
    A1 = 3.2345
    A2 = 235.88
    A3 = 0.038371
    A4 = 3.8068e-6
    A5 = 1.1832e-10
    A6 = 2.3713
    return (1e-16 * A1 * np.log(A2 / E + A6)) / (1.0 + A3 * E + A4 * E ** 3.5 + A5 * E ** 5.4)  # cm^2


def js_sigma_cx_ng1_q1(E, n1):
    """Electron capture cross section for

    .. math::

        H^{+} + H(n) --> H + H^+ , n>1

    Notes
    ---------
    Section 2.3.2 of Janev & Smith, NF 1993. 
    """
    assert n1 > 1
    if n1 == 2:
        A1 = 0.92750
        A2 = 6504e3
        A3 = 1.3405e-2
        A4 = 20.699
    elif n1 == 3:
        A1 = 0.37271
        A2 = 2.7645e6
        A3 = 1.5720e-3
        A4 = 1.4857e3
    else:  # n1>=4
        A1 = 0.21336
        A2 = 1e10
        A3 = 1.8184e-3
        A4 = 1.3426e6

    Ew = E * n1 ** 2
    return n1 ** 4 * (1e-16 * A1 * np.log(A2 / Ew + A4)) / (1.0 + A3 * Ew + 3.0842e-6 * Ew ** 3.5 + 1.1832e-10 * Ew ** 5.4)  # cm^2


def js_sigma_cx_n1_q2(E):
    """Electron capture cross section for

    .. math::

        He^{2+} + H(1s) --> He^+ + H^+

    Notes
    ---------
    Section 3.3.1 of Janev & Smith, NF 1993. 
    """
    A1 = 17.438
    A2 = 2.1263
    A3 = 2.1401e-3
    A4 = 1.6498
    A5 = 2.6259e-6
    A6 = 2.4226e-11
    A7 = 15.665
    A8 = 7.9193
    A9 = -4.4053

    return 1e-16 * A1 * ((np.exp(-A2 / E) / (1.0 + A3 * E ** A4 + A5 * E ** 3.5 + A6 * E ** 5.4)) + (A7 * np.exp(-A8 * E)) / (E ** A9))


def js_sigma_cx_n2_q2(E):
    """Electron capture cross section for

    .. math::
    
        He^{2+} + H(n=2) --> He^+ + H^+

    Notes
    ---------
    Section 3.3.2 of Janev & Smith, NF 1993. 
    """
    A1 = 88.508
    A2 = 0.78429
    A3 = 3.2903e-2
    A4 = 1.7635
    A5 = 7.3265e-5
    A6 = 1.4418e-8
    A7 = 0.80478
    A8 = 0.22349
    A9 = -0.68604
    return (
        1e-16 * A1 * ((np.exp(-A2 / E)) / (1.0 + A3 * E ** A4 + A5 * E ** 3.5 + A6 * E ** 5.4) + (A7 * np.exp(-A8 * E)) / (E ** A9))
    )  # cm^2


def js_sigma_cx_ng2_q2(E, n1):
    """Electron capture cross section for

    .. math::
        He^{2+} + H*(n) --> He^+ + H^+ , n>2

    Notes
    ---------
    Section 3.2.3 of Janev & Smith, NF 1993. 
    """
    A1 = 2.0032e2
    A2 = 1.4591
    A3 = 2.0384e-4
    A4 = 2e-9

    Ew = E * n1 ** 2
    return (
        n1 ** 4
        * 7.04e-16
        * A1
        * (1.0 - np.exp(-(4.0 / 3.0 * A1) * (1.0 + Ew ** A2 + A3 * Ew ** 3.5 + A4 * Ew ** 5.4)))
        / (1.0 + Ew ** A2 + A3 * Ew ** 3.5 + A4 * Ew ** 5.4)
    )  # cm^2


def js_sigma_cx_n1_q4(E):
    """Electron capture cross section for

    .. math::

        Be^{4+} + H(1s) --> Be^{3+} + H^+

    Notes
    ---------
    Section 4.3.1 of Janev & Smith, NF 1993. 
    """
    A1 = 19.952
    A2 = 0.20036
    A3 = 1.7295e-4
    A4 = 3.6844e-11
    A5 = 5.0411
    A6 = 2.4689e-8
    A7 = 4.0761
    A8 = 0.88093
    A9 = 0.94361
    A10 = 0.14205
    A11 = -0.42973
    return (
        1e-16
        * A1
        * ((np.exp(-A2 / (E ** A8))) / (1.0 + A3 * (E ** 2) + A4 * (E ** A5) + A6 * (E ** A7)) + (A9 * np.exp(-A10 * E)) / (E ** A11))
    )  # cm^2


def js_sigma_cx_n1_q5(E):
    """Electron capture cross section for

    .. math::

        B^{5+} + H(1s) --> B^{4+} + H^+

    Notes
    ---------
    Section 4.3.2 of Janev & Smith, NF 1993. 
    """
    A1 = 31.226
    A2 = 1.1442
    A3 = 4.8372e-8
    A4 = 3.0961e-10
    A5 = 4.7205
    A6 = 6.2844e-7
    A7 = 3.1297
    A8 = 0.12556
    A9 = 0.30098
    A10 = 5.9607e-2
    A11 = -0.57923
    return (
        1e-16
        * A1
        * ((np.exp(-A2 / (E ** A8))) / (1.0 + A3 * (E ** 2) + A4 * (E ** A5) + A6 * (E ** A7)) + (A9 * np.exp(-A10 * E)) / (E ** A11))
    )  # cm^2


def js_sigma_cx_n1_q6(E):
    """Electron capture cross section for

    .. math::
    
        C^{6+} + H(1s) --> C^{5+} + H^+

    Notes
    ---------
    Section 4.3.3 of Janev & Smith, NF 1993. 
    """
    A1 = 418.18
    A2 = 2.1585
    A3 = 3.4808e-4
    A4 = 5.3333e-9
    A5 = 4.6556
    A6 = 0.33755
    A7 = 0.81736
    A8 = 0.27874
    A9 = 1.8003e-6
    A10 = 7.1033e-2
    A11 = 0.53261
    return (
        1e-16
        * A1
        * ((np.exp(-A2 / (E ** A8))) / (1.0 + A3 * (E ** 2) + A4 * (E ** A5) + A6 * (E ** A7)) + (A9 * np.exp(-A10 * E)) / (E ** A11))
    )  # cm^2


def js_sigma_cx_n1_q8(E):
    """Electron capture cross section for

    .. math::

        O^{8+} + H(1s) --> O^{7+} + H^+

    Notes
    ---------
    Section 4.3.4 of Janev & Smith, NF 1993. 
    """
    A1 = 1244.44
    A2 = 249.36
    A3 = 30.892
    A4 = 9.0159e-4
    A5 = 7.7885e-3
    A6 = -0.71309
    A7 = 3.2918e3
    A8 = -2.7541
    return 1e-16 * A1 * ((np.exp(-A2 / E) * np.log(1.0 + A3 * E)) / E + (A4 * np.exp(-A5 * E)) / (E ** A6 + A7 * E ** A8))  # cm^2


def js_sigma_cx_n1_qg8(E, q):
    """Electron capture cross section for

    .. math::

        A^{q+} + H(1s) --> A^{(q-1)+} + H^+,   q>8

    Notes
    ---------
    Section 4.3.5, p.172, of Janev & Smith, NF 1993. 
    """
    Ew = E / q ** (3.0 / 7.0)
    A1 = 0.73362
    A2 = 2.9391e4
    A3 = 41.8648
    A4 = 7.1023e-3
    A5 = 3.4749e-6
    A6 = 1.1832e-10
    return q * (1e-16 * A1 * np.log(A2 / Ew + A3)) / (1 + A4 * Ew + A5 * Ew ** 3.5 + A6 * Ew ** 5.4)  # cm^2


def js_sigma_cx_ng1_qg3(E, n1, q):
    """Electron capture cross section for

    .. math::

        A^{q+} + H^*(n) --> A^{(q-1)+}+H^+ , q>3

    Notes
    ---------
    Section 4.3.6, p.174, of Janev & Smith, NF 1993. 
    """
    A = 1.507e5
    B = 1.974e-5
    Ew = E * n1 ** 2 / q ** 0.5
    return (
        q * n1 ** 4 * 7.04e-16 * A / (Ew ** 3.5 * (1 + B * Ew ** 2)) * (1 - np.exp((-2.0 * Ew ** 3.5 * (1 + B * Ew ** 2)) / (3.0 * A)))
    )  # cm^2


def js_sigma(E, q, n1, n2=None, type='cx'):
    """Cross sections for collisional processes between beam neutrals and highly-charged 
    ions, from Janev & Smith 1993.

    Parameters
    ----------
    E : float
        Normalized beam energy [keV/amu]
    q : int
        Impurity charge before interaction (interacting ion is :math:`A^{q+}`)
    n1 : int
        Principal quantum number of beam hydrogen.
    n2: int
        Principal quantum number of excited. This may not be needed for some transitions (if so, leave to None).
    type : str
        Type of interaction. Possible choices:
        {'exc','ioniz','cx'}
        where 'cx' refers to electron capture / charge exchange.

    Returns
    -------
    sigma : float
        Cross section of selected process, in [:math:`cm^2`] units.

    Notes
    -----
    See comments in Janev & Smith 1993 for uncertainty estimates.
    """

    if type == 'exc':
        # p.124 - 146
        raise ValueError('Not implemented yet')

    elif type == 'ioniz':
        # p.150-160
        if n1 == 1 and q == 8:  # O-like
            sigma = js_sigma_ioniz_n1_q8(E)
        else:
            raise ValueError('Not implemented yet')

    elif type == 'cx':  # electron capture
        if n1 == 1 and q == 1:
            sigma = js_sigma_cx_n1_q1(E)

        elif n1 > 1 and q == 1:
            sigma = js_sigma_cx_ng1_q1(E, n1)

        elif n1 == 1 and q == 2:
            sigma = js_sigma_cx_n1_q2(E)

        elif n1 == 2 and q == 2:
            sigma = js_sigma_cx_n2_q2(E)

        elif n1 > 2 and q == 2:
            sigma = js_sigma_cx_ng2_q2(E, n1)

        elif q == 3:
            # NO RATES AVAILABLE
            # Substitute with He-like ones multiplied by 3/2, using the linear scaling seen for q>8
            if n1 == 1:
                sigma = 1.5 * js_sigma_cx_n1_q2(E)
            elif n1 == 2:
                sigma = 1.5 * js_sigma_cx_n2_q2(E)
            else:
                sigma = 1.5 * js_sigma_cx_ng2_q2(E, n1)

        elif n1 == 1 and q == 4:  # Be-like
            sigma = js_sigma_cx_n1_q4(E)

        if n1 == 1 and q == 5:  # B-like
            sigma = js_sigma_cx_n1_q5(E)

        if n1 == 1 and q == 6:  # C-like
            sigma = js_sigma_cx_n1_q6(E)

        if n1 == 1 and q == 7:  # N-like
            # NO RATES AVAILABLE!
            # Substitute with C-like ones multiplied by 7/6, using the linear scaling seen for q>8
            sigma = (7.0 / 6.0) * js_sigma_cx_n1_q8(E)

        if n1 == 1 and q == 8:  # O-like
            sigma = js_sigma_cx_n1_q8(E)

        if n1 == 1 and q > 8:  # higher ionization stages, n1=1
            sigma = js_sigma_cx_n1_qg8(E, q)

        elif n1 > 1 and q > 3:  # for all excited states of q>3
            sigma = js_sigma_cx_ng1_qg3(E, n1, q)

    else:
        raise ValueError('Unrecognized type of interaction')

    return sigma




def plot_js_sigma(q=18):
    '''Plot/check sensitivity of JS cross sections to beam energy.
    NB: cross section is taken to only depend on partially-screened ion charge
    '''
    Ebeam = np.geomspace(10, 1e2, 1000)  # keV/amu

    sigma = np.array([js_sigma(E, q, n1=1, type='cx') for E in Ebeam])
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    ax[0].loglog(1e3 * Ebeam / (q ** (3.0 / 7.0)), sigma / q, '*-')
    ax[0].set_xlabel(r'Scaled Energy ($E/q^{3/7}$) [eV/amu]', fontsize=18)
    ax[0].set_ylabel(r'Scaled Cross Section ($\sigma_{cx}/q$) [$cm^2$]', fontsize=18)
    ax[0].grid(True, which='both')
    ax[0].set_xlim([np.min(1e3 * Ebeam / (q ** (3.0 / 7.0))), np.max(1e3 * Ebeam / (q ** (3.0 / 7.0)))])

    ax[1].loglog(Ebeam, sigma, '*-')
    ax[1].set_xlabel(r'$E$ [keV/amu]', fontsize=18)
    ax[1].set_ylabel(r'$\sigma_{cx}$ [$cm^2$]', fontsize=18)
    ax[1].grid(True, which='both')
    ax[1].set_xlim([np.min(Ebeam), np.max(Ebeam)])
    plt.suptitle(r'$A^{q+}$ + $H^*(n=1)$ --> $A^{(q-1)+}$+$H^+$ , $q>8$', fontsize=20)

    # n1=2 excited state
    n1 = 2
    sigma = np.array([js_sigma(E, q, n1=n1, type='cx') for E in Ebeam])
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    ax[0].loglog(1e3 * Ebeam * n1 ** 2 / (q ** 0.5), sigma / (q * n1 ** 4), '*-')
    ax[0].set_xlabel(r'Scaled energy ($E n^2 / q^{0.5}$) [eV/amu]', fontsize=18)
    ax[0].set_ylabel(r'Scaled Cross Section ($\sigma_{cx}/(q n^4)$) [$cm^2$]', fontsize=18)
    ax[0].grid(True, which='both')
    ax[0].set_xlim([np.min(1e3 * Ebeam * n1 ** 2 / (q ** 0.5)), np.max(1e3 * Ebeam * n1 ** 2 / (q ** 0.5))])

    ax[1].loglog(Ebeam, sigma, '*-')
    ax[1].set_xlabel(r'$E$ [keV/amu]', fontsize=18)
    ax[1].set_ylabel(r'$\sigma_{cx}$ [$cm^2$]', fontsize=18)
    ax[1].grid(True, which='both')
    ax[1].set_xlim([np.min(Ebeam), np.max(Ebeam)])
    plt.suptitle(r'$A^{q+}$ + $H^*(n=2)$ --> $A^{(q-1)+}$+$H^+$ , $q>3$', fontsize=20)
