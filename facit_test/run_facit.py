#import omfit_classes
from omfit_classes import omfit_gacode

from scipy.constants import e,m_u
import aurora
import numpy as np
import matplotlib.pylab as plt
plt.ioff()


input_gacode = omfit_gacode.OMFITinputgacode('input.gacode')
ion_num = 2
rmin = input_gacode['rmin']
a = input_gacode['rmin'][-1]
r_a  = rmin / a
 

vtor = input_gacode['omega0'] * (input_gacode['rmaj']+input_gacode['rmin'])
Machi =  np.sqrt(2*m_u/e* vtor **2/(2*  input_gacode['Ti_1']*1e3 ))

ion = input_gacode['IONS'][ion_num]
name, Zimp, Aimp, _ = ion
 
 
R, Z = input_gacode.rz_geometry()
TeovTi = input_gacode['Te']/input_gacode['Ti_1']
Zeff = 1
rotation_model = 2
fct = aurora.FACIT(
        r_a ,
        Zimp, 
        Aimp,
        1, 
        2, 
        input_gacode['Ti_1'] * 1e3,
        input_gacode['ni_1'] * 1e19,
        input_gacode['ni_1']*1e10,
        Machi,
        Zeff ,
        np.gradient(input_gacode['Ti_1']* 1e3, rmin),
        np.gradient(input_gacode['ni_1']* 1e19, rmin),
        0,
        rmin / input_gacode['rmaj'],
        abs(input_gacode['BT_EXP']),
        input_gacode['rmaj'][0],
        np.abs(input_gacode['q']),
        rotation_model=rotation_model,
        Te_Ti= TeovTi,
        RV=R.T,
        ZV=Z.T,
        fsaout=False
    )

plt.subplot(121)
plt.semilogy(r_a, fct.Dz)
plt.subplot(122)
plt.plot(r_a, fct.Vconv)
plt.show()
