'''Aurora tools to read SOLPS results and extract atomic neutral density from EIRENE output.
These enable examination of charge exchange recombination and radiation due to the interaction 
of heavy ions and thermal neutrals.
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

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata, interp1d
import matplotlib as mpl
import matplotlib.tri as tri
from matplotlib import collections as mc
from heapq import nsmallest
import warnings
from scipy import constants
from . import plot_tools
from . import coords


class solps_case:
    '''Read SOLPS output, either from 
    
    #. output files on disk 

    #. an MDS+ tree

    If arguments are provided with no keys, it is assumed that paths to b2fstate andd b2fgmtry
    are being provided (option #1). If keyword arguments are provided instead, we first check
    whether `b2fstate_path` and `b2fgmtry_path` are being given (option #1) or `solps_id` is 
    given to load results from MDS+ (option #2). Other keyword arguments can be provided to
    specify the MDS+ server and tree to use (defaults are for AUG), and to provide a gEQDSK file.
    
    Parameters
    ----------
    b2fstate_path : str (option #1)
        Path to SOLPS b2fstate output file. 
    b2fgmtry_path : str (option #1)
        Path to SOLPS b2fgmtry output file.
    solps_id : str or int (option #2)
        Integer identifying the SOLPS run/shot to load from MDS+. 
    server : str, optional (option #2)
        MDS+ server to load SOLPS results. Default is 'solps-mdsplus.aug.ipp.mpg.de:8001'.
    tree : str (option #2)
        Name of MDS+ tree to load data from. Default is 'solps'.
    geqdsk : str, optional (option #1 and #2)
        Path to the geqdsk to load from disk.
        Users may also directly provide an instance of the 
        `omfit_classes.omfit_geqdsk.OMFITgeqdsk` class that contains the processed gEQDSK file.
        If not provided, the code tries to reconstruct a geqdsk based on the experiment name and 
        time of analysis stored in the SOLPS output. If set to None, no geqdsk is loaded and 
        functionality related to the geqdsk is not used.

    Notes
    -----
    The b2fstate and b2fgmtry parser expects the filenames to be simply named "b2fstate" and "b2fgmtry",
    or else it may not recognize fields appropriately.

    Minimal Working Example
    -----------------------
    import aurora
    so = aurora.solps_case(141349) # load case 141349 from AUG MDS+; must have access to the server!
    so.plot2d_b2(so.data('ne'))

    '''
    def __init__(self, *args, **kwargs):
        if len(args):
            # if arguments are provided with no keys, assume they are paths to b2fstate andd b2fgmtry
            self.form = 'files'
            self.b2fstate_path = str(args[0])
            self.b2fgmtry_path = str(args[1])
            if len(args)>2:
                self.geqdsk_path = args[2] # can be path, or geqdsk dict-like
        else:
            if 'b2fstate_path' in kwargs or 'b2fgmtry_path' in kwargs:
                self.form = 'files'
                self.b2fstate_path = str(kwargs['b2fstate_path'])
                self.b2fgmtry_path = str(kwargs['b2fgmtry_path'])
            elif 'solps_id' in kwargs:
                # load from MDS+
                self.form = 'mdsplus'
                self.solps_id = kwargs['solps_id']                                       
                self.server = 'solps-mdsplus.aug.ipp.mpg.de:8001' if 'server' not in kwargs else kwargs['server']
                self.tree = 'solps'  if 'tree' not in kwargs else kwargs['tree']
            else:
                raise ValueError('Insufficient information passed as (keyword) arguments to read SOLPS results')

        if self.form=='files':
            from omfit_classes import omfit_solps
            self.b2fstate = omfit_solps.OMFITsolps(self.b2fstate_path)
            self.b2fgmtry = omfit_solps.OMFITsolps(self.b2fgmtry_path)

            self.nx,self.ny = self.b2fgmtry['nx,ny']
            
            # (0,:,:): lower left corner, (1,:,:): lower right corner
            # (2,:,:): upper left corner, (3,:,:): upper right corner.
            
            self.crx = self.b2fgmtry['crx'].reshape(4,self.ny+2,self.nx+2)  # horizontal 
            self.cry = self.b2fgmtry['cry'].reshape(4,self.ny+2,self.nx+2)  # vertical

            # now, load data arrays
            self.load_data() #(P_idxs=self.P_idxs, R_idxs=self.R_idxs)  # TODO: enable subselection of regions
        
        elif self.form=='mdsplus':
            # load variable name map to MDS+ tree
            self.mdsmap = get_mdsmap()

        # figure out if single or double null
        self.DN = self.data('cry').shape[2]%4==0

        # is the definition of these units robust?
        self.unit_p = (self.data('crx').shape[2]-4 if self.DN else self.data('crx').shape[2])//4
        self.unit_r = (self.data('crx').shape[1]-2)//2

        if self.DN:
            # TODO: double null generalization still incomplete!

            # Obtain indices for chosen radial regions
            _R_idxs = np.array([],dtype=int)
            _R_idxs = np.concatenate((_R_idxs, np.arange(self.unit_r)))  # PFR and core
            self.R_idxs = np.concatenate((_R_idxs, np.arange(self.unit_r,2*self.unit_r))) # open-SOL

            # obtain indices for chosen poloidal regions
            _P_idxs = np.array([],dtype=int)
            _P_idxs = np.concatenate((_P_idxs, np.arange(self.unit_p+1)))  # Inner PFR
            _P_idxs = np.concatenate((_P_idxs, np.arange(self.unit_p+1,3*self.unit_p+1)))  # core/open SOL
            self.P_idxs = np.concatenate((_P_idxs, np.arange(3*self.unit_p+1,4*self.unit_p+2)))  # outer PFR

        else:  # upper or lower single null
            # Obtain indices for chosen radial regions
            _R_idxs = np.array([],dtype=int)
            _R_idxs = np.concatenate((_R_idxs, np.arange(self.unit_r+1)))  # PFR and core
            self.R_idxs = np.concatenate((_R_idxs, np.arange(self.unit_r+1,2*self.unit_r+2))) # open-SOL

            # obtain indices for chosen poloidal regions
            _P_idxs = np.array([],dtype=int)
            _P_idxs = np.concatenate((_P_idxs, np.arange(self.unit_p+1)))  # Inner PFR
            _P_idxs = np.concatenate((_P_idxs, np.arange(self.unit_p+1,3*self.unit_p+1)))  # core/open SOL
            self.P_idxs = np.concatenate((_P_idxs, np.arange(3*self.unit_p+1,4*self.unit_p+2)))  # outer PFR

        # set zero densities equal to min to avoid log issues
        #self.quants['nn'][self.quants['nn']==0.0] = nsmallest(2,np.unique(self.quants['nn'].flatten()))[1]

        # identify species (both B2 and EIRENE):
        self.species_id()

        # indices of neutral species in B2 output
        self.neutral_idxs = np.array([ii for ii in list(self.b2_species.keys()) if self.b2_species[ii]['Z']==0])
        
        # SOLPS does not enforce that neutral stage densities are from EIRENE in output. Do this here:
        #self.data('na')[neutral_idx,:,:] = self.data('dab2')
        
        # geqdsk offers a description of the magnetic equilibrium; only used here for visualization + postprocessing
        if 'geqdsk' in kwargs:
            if kwargs['geqdsk'] is None:
                # user explicitly requested not to use geqdsk functionality
                self.geqdsk = None
            else:
                from omfit_classes import omfit_eqdsk
                if isinstance(kwargs['geqdsk'], str):
                    # load geqdsk file
                    self.geqdsk = omfit_eqdsk.OMFITgeqdsk(kwargs['geqdsk'])
                else:
                    # assume geqdsk was already loaded via OMFITgeqdsk
                    self.geqdsk = kwargs['geqdsk']
        else:
            from omfit_classes import omfit_eqdsk
            # try to find gEQDSK and load it
            _gfile_path = None if self.form=='mdsplus' else self.find_gfile()
            if _gfile_path is not None:
                self.geqdsk = omfit_eqdsk.OMFITgeqdsk(_gfile_path)
            else:
                # attempt to reconstruct geqdsk from device MDS+ server
                try:
                    if 'AUG' in self.data('exp')[0]:
                        self.geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_aug_sfutils(
                            int(self.data('shot')), float(self.data('time')), 'EQI')
                    else:
                        self.geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                            device=self.data('exp'), shot=int(self.data('shot')),
                            time=float(self.data('time'))*1e3) # time in ms
                except Exception as e:
                    # geqdsk could not be fetched
                    pass
                
        # if user provided path to b2fplasmf file, load that too
        if 'b2fplasmf_path' in kwargs:
            self.b2fplasmf_path = str(kwargs['b2fplasmf_path'])
            self.b2fplasmf = omfit_solps.OMFITsolps(self.b2fstate_path)


    def data(self, varname):
        '''Fetch data either from files or MDS+ tree.

        Parameters
        ----------
        varname : str
            Name of SOLPS variable, e.g. ne,te,ti,dab2,etc. 
        '''
        if hasattr(self, varname):
            return getattr(self, varname)
        elif self.form=='mdsplus':
            # try fetching from MDS+
            from omfit_classes import omfit_mds
            # cache quantities fetched from MDS+ to increase speed
            setattr(self, varname, omfit_mds.OMFITmdsValue(self.server, self.tree, self.solps_id,
                                                           TDI = self.mdsmap[varname]).data())
            return getattr(self, varname)
        else:
            raise ValueError(f'Could not fetch variable {varname}')

    def load_data(self, fields=None, P_idxs=None, R_idxs=None,
                  Rmin=None, Rmax=None, Pmin=None, Pmax=None):
        '''Load SOLPS output for each of the needed quantities

        Parameters
        -----------------
        fields : list or array
            List of fields to fetch from SOLPS output. If left to None, by default uses
            ['ne','Te','nn','Tn','nm','Tm','Ti']
        P_idxs : list or array
            Poloidal indices to load.
        R_idxs : list or array
            Radial indices to load.
        Rmin : int or None.
            Minimum major radius index to load, if R_idxs is not given
        Rmax : int or None
            Maximum major radius index to load, if R_idxs is not given
        Pmin : int or None
            Minimum poloidal index to load, if P_idxs is not given
        Pmax : int or None
            Maximum poloidal index to load, if P_idxs is not given

        '''
        if P_idxs is None:
            if Pmax is None: Pmax = self.nx
            if Pmin is None: Pmin = 0
            P_idxs = np.arange(Pmin,Pmax)

        if R_idxs is None:
            if Rmax is None: Rmax = self.ny
            if Rmin is None: Rmin = 0
            R_idxs = np.arange(Rmin,Rmax)

        # eliminate end (buffer) points of grid
        self.cr = np.mean(self.data('crx'), axis=0)[1:-1,1:-1][R_idxs,:][:,P_idxs]
        self.cz = np.mean(self.data('cry'), axis=0)[1:-1,1:-1][R_idxs,:][:,P_idxs]

        self.ne = self.b2fstate['ne'][1:-1,1:-1][R_idxs,:][:,P_idxs] # m^-3
        self.te = self.b2fstate['te'][1:-1,1:-1][R_idxs,:][:,P_idxs]/constants.e # eV
        self.ti = self.b2fstate['ti'][1:-1,1:-1][R_idxs,:][:,P_idxs]/constants.e # eV

        # density of all atomic/ionic species.
        self.na = self.b2fstate['na'][:,R_idxs,:][:,:,P_idxs] # m^-3  # all density components

        try:
            self.fort44 = self.load_fort44()

            # assume that EIRENE neutral model is used:
            nn44 = self.fort44['dab2'][:,:,0].T
            nn44[nn44==0.0] = nsmallest(2,np.unique(nn44.flatten()))[1]
            self.dab2 = nn44[R_idxs,:][:,P_idxs] # m^-3   # only neutral component
            self.tab2 = self.fort44['tab2'][:,:,0].T[R_idxs,:][:,P_idxs]/constants.e # eV
            
            self.dmb2 = self.fort44['dmb2'][:,:,0].T[R_idxs,:][:,P_idxs] # D molecular density
            self.tmb2 = self.fort44['tmb2'][:,:,0].T[R_idxs,:][:,P_idxs] # D molecular temperature
        except Exception:
            warnings.warn('Could not load fort.44 file')

        try:
            self.fort46 = self.load_fort46()
        except Exception:
            warnings.warn('Could not load fort.46 file')

        try:
            # EIRENE nodes and triangulation
            self.xnodes, self.ynodes, self.triangles = self.load_eirene_mesh()
        except Exception:
            warnings.warn('Could not load fort.33 and/or fort.34 files')
            
    def species_id(self):
        '''Identify species included in SOLPS run, both for B2 and EIRENE quantities.
        This is only designed to work in the "full" data format.
        '''
        # nuclear charge (not ionization state)
        #b2_zn = self.b2fstate['zn'] if self.form=='files' else self.data('zn')

        # nuclear masses (2 for H/D/T, 4 for He, etc.)
        b2_am = self.b2fstate['am'] if self.form=='files' else self.data('am') 
            
        # atom charge
        if self.form=='files' and (not np.array_equal(self.b2fstate['zamin'], self.b2fstate['zamax'])):
            # using charge state bundling
            raise ValueError('Detected charge state bundling; analysis not yet implemented')
        b2_za = self.b2fstate['zamin'] if self.form=='files' else self.data('za')

        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element, toRoman
        
        self.b2_species = {}
        for ii,(Zi,An) in enumerate(zip(b2_za,b2_am)):
            # get name of atomic species from charge and atomic mass number
            out = atomic_element(Z_ion=int(Zi), A=int(An))  # requires integer values
            _spec = list(out.keys())[0]
            cs_str = f' {toRoman(int(Zi)+1)}'
            self.b2_species[ii] = {'symbol': out[_spec]['symbol']+cs_str, 'Z':Zi, 'A':An}

        if hasattr(self, 'fort44'): # loaded EIRENE data
            self.eirene_species = {'atm':{}, 'mol':{}, 'ion':{}}
            _atm = [spec for spec in self.fort44['species'] if ((not any(map(str.isdigit, spec)))&('+' not in spec)) ]
            for ss,atm in enumerate(_atm): self.eirene_species['atm'][ss] = atm
            _mol = [spec for spec in self.fort44['species'] if ((any(map(str.isdigit, spec)))&('+' not in spec))]
            for ss,mol in enumerate(_mol): self.eirene_species['mol'][ss] = mol
            _ion = [spec for spec in self.fort44['species'] if '+'  in spec]
            for ss,ion in enumerate(_ion): self.eirene_species['ion'][ss] = ion

    def load_mesh_extra(self):
        '''Load the mesh.extra file.
        '''
        with open(os.path.dirname(self.b2fgmtry_path)+os.sep+'mesh.extra', 'r') as f:
            contents = f.readlines()

        mesh_extra = np.zeros((len(contents),4))
        ii=0
        while ii<len(contents):
            mesh_extra[ii,:] = [float(val) for val in contents[ii].split()]
            ii+=1

        return mesh_extra

    def load_fort44(self):
        '''Load result from one of the fort.44 file with EIRENE output on B2 grid.

        Returns
        -------
        out : dict
            Dictionary containing a subdictionary with keys for each loaded field. 
        '''
        out = {}
            
        out = {}
        # load each of these files into dictionary structures
        with open(os.path.dirname(self.b2fstate_path)+os.sep+'fort.44', 'r') as f:
            contents = f.readlines()

        NDX = out['NDX'] = int(contents[0].split()[0])
        NDY = out['NDY']= int(contents[0].split()[1])
        #out['DIM']=NDX*NDY
        
        NATM = out['NATM'] = int(contents[1].split()[0][0])
        NMOL = out['NMOL'] = int(contents[1].split()[1][0])
        NION = out['NION'] = int(contents[1].split()[2][0])

        ii=2
        out['species']=[]
        while not contents[ii].startswith('*eirene'):
            if not contents[ii].split()[0].isnumeric():
                out['species'].append(contents[ii].split()[0])
            ii+=1                    

        while ii<len(contents):
            if contents[ii].startswith('*eirene'):
                key = ''.join(contents[ii].split()[3:-3])
                dim = int(contents[ii].split()[-1])
                out[key] = []
            elif not contents[ii].split()[0][0].isalpha():
                [out[key].append(float(val)) for val in contents[ii].strip().split()]
            ii+=1

        # hidden line at the end of 'edissml' field gives values of NLIM,NSTS,NSTRA
        NLIM,NSTS,NSTRA = [int(val) for val in out['edissml'][-3:]]
        out['NLIM'] = NLIM
        NSTS=out['NSTS'] = NSTS
        NSTRA=out['NSTRA'] = NSTRA
        del out['edissml'][-3:]

        # number of source strata
        _NSTRA= np.array(out['pdena_int']).reshape(NATM,-1, order='F').shape[1]-1
        assert NSTRA==_NSTRA   #check
        # number of segments for resolved tallies on non-standard surfaces
        NCL=out['NCL'] = len(out['sarea_res'])
        # number of plasma background species
        NPLS=out['NPLS'] = np.array(out['wlpump_res(P)']).reshape(NCL,-1,order='F').shape[1]
        # number of “non-standard surfaces” (plasma boundaries)
        _NSTS = int((len(out['eirdiag'])-1)/5)
        assert NSTS==_NSTS #check
        # number of "additional surfaces" (physical walls)
        _NLIM=out['NLIM'] = len(out['isrftype'])-NSTS
        assert NLIM==_NLIM #check
        
        # load variable descriptors for fort.44
        fort44_info = get_fort44_info(NDX,NDY,NATM,NMOL,NION,NSTRA,NCL,NPLS,NSTS,NLIM)

        for key in out:
            if key in ['species','NDX','NDY','NATM','NMOL','NION','NSTRA','NCL','NPLS','NSTS','NLIM']:
                continue
            # find appropriate shape from fort44_info dictionary
            try:
                out[key] = np.array(out[key]).reshape(*fort44_info[key][1],order='F')
            except:
                print(f'Variable {key} in fort.44 could not be parsed, likely to be new in SOLPS-ITER.')

        return out

    def load_fort46(self):
        '''Load result from fort.46 file with EIRENE output on EIRENE grid.

        Returns
        -------
        out : dict
            Dictionary for each loaded file containing a subdictionary with keys for each loaded field from each file. 
        '''
        out = {}
            
        out = {}
        # load each of these files into dictionary structures
        with open(os.path.dirname(self.b2fstate_path)+os.sep+'fort.46', 'r') as f:
            contents = f.readlines()

        out['NTRII']=contents[0].split()[0]

        NATM = out['NATM'] = int(contents[1].split()[0][0])
        NMOL = out['NMOL'] = int(contents[1].split()[1][0])
        NION = out['NION'] = int(contents[1].split()[2][0])

        ii=2
        out['species']=[]
        while not contents[ii].startswith('*eirene'):
            if not contents[ii].split()[0].isnumeric():
                out['species'].append(contents[ii].split()[0])
            ii+=1                    

        while ii<len(contents):
            if contents[ii].startswith('*eirene'):
                key = ''.join(contents[ii].split()[3:-3])
                dim = int(contents[ii].split()[-1])
                out[key] = []
            elif not contents[ii].split()[0][0].isalpha():
                [out[key].append(float(val)) for val in contents[ii].strip().split()]
            ii+=1

        for key in out:

            # set to the right shape, depending on whether quantity is atomic, molecular or ionic
            if key.endswith('a'): num=NATM
            elif key.endswith('m'): num=NMOL
            elif key.endswith('i'): num=NION
            else: num=1
            if key in ['species','NTRII']:
                continue
            out[key] = np.array(out[key]).reshape(-1,num,order='F')

        return out
    
    def load_eirene_mesh(self):
        '''Load EIRENE nodes from the fort.33 file and triangulation from the fort.34 file
        '''
        nodes=np.fromfile(os.path.dirname(self.b2fgmtry_path)+os.sep+'fort.33',sep=' ')
        n=int(nodes[0])
        xnodes=nodes[1:n+1]/100  # cm -->m
        ynodes=nodes[n+1:]/100
        
        # EIRENE triangulation
        triangles = np.loadtxt(os.path.dirname(self.b2fgmtry_path)+os.sep+'fort.34',
                               skiprows=1, usecols=(1,2,3))

        return xnodes, ynodes, triangles-1  # -1 for python indexing

    def plot_wall_geometry(self):
        '''Method to plot vessel wall segment geometry from wall_geometry field in fort.44 file'''
        
        out = self.load_fort44()
        wall_geometry = out['wall_geometry']
        
        Wall_Seg = []
        RR = wall_geometry[0::2]
        ZZ = wall_geometry[1::2]
        NLIM = out['NLIM']
        
        for i in range(0,NLIM):
            line = [(RR[2*i],ZZ[2*i]),(RR[2*i+1],ZZ[2*i+1])]
            Wall_Seg.append(line)
            
        Wall_Collection = mc.LineCollection(Wall_Seg,colors='b',linewidth=2)
        
        wallfig, wallax = plt.subplots()
        
        wallax.add_collection(Wall_Collection)
        wallax.set_xlim(RR.min()-0.05,RR.max()+0.05)
        wallax.set_ylim(ZZ.min()-0.05,ZZ.max()+0.05)
        wallax.set_xlabel('R [m]')
        wallax.set_ylabel('Z [m]')
        wallax.set_aspect('equal')
        
        self.WS=Wall_Seg
        self.WC=Wall_Collection

    def get_b2_patches(self):
        '''Get polygons describing B2 grid as a mp.collections.PatchCollection object.
        '''
        xx = self.data('crx').transpose(2,1,0)
        yy = self.data('cry').transpose(2,1,0)
        NY = int(self.data('ny'))
        NX = int(self.data('nx'))

        if self.form=='files':
            # eliminate boundary cells
            xx = xx[1:-1,1:-1,:]
            yy = yy[1:-1,1:-1,:]

        patches = []
        for iy in np.arange(0,NY):
            for ix in np.arange(0,NX):
                rr = np.atleast_2d(xx[ix,iy,[0,1,3,2]]).T
                zz = np.atleast_2d(yy[ix,iy,[0,1,3,2]]).T
                patches.append( mpl.patches.Polygon(np.hstack((rr,zz)), True, linewidth=3) )

        # collect all patches
        return mpl.collections.PatchCollection(patches, False, fc='w', edgecolor='k', linewidth=0.1)

    def plot2d_b2(self, vals, ax=None, scale='log', label='', lb=None, ub=None, **kwargs):
        '''Method to plot 2D fields on B2 grids. 
        Colorbars are set to be manually adjustable, allowing variable image saturation.

        Parameters
        ----------
        vals : array (self.data('ny'), self.data('nx'))
            Data array for a variable of interest.
        ax : matplotlib Axes instance
            Axes on which to plot. If left to None, a new figure is created.
        scale : str
            Choice of 'linear','log' and 'symlog' for matplotlib.colors.
        label : str
            Label to set on the colorbar. No label by default.
        lb : float
            Lower bound for colorbar. If left to None, the minimum value in `vals` is used.
        ub : float
            Upper bound for colorbar. If left to None, the maximum value in `vals` is used.
        kwargs
            Additional keyword arguments passed to the `PatchCollection` class.
        '''
        if ax is None:
            fig,ax = plt.subplots(1,figsize=(9, 11))

        # get polygons describing B2 grid
        p = self.get_b2_patches()
        
        if self.form=='files' and np.prod(vals.shape)==self.data('crx').shape[1]*self.data('crx').shape[2]:
            vals = vals[1:-1,1:-1]

        # fill patches with values
        _vals = vals.flatten()

        # Avoid zeros that may derive from low Monte Carlo statistics
        if np.any(_vals==0): _vals[_vals==0.0] = nsmallest(2,np.unique(_vals))[1]
        
        p.set_array(np.array(_vals))

        if lb is None: lb = np.min(_vals)
        if ub is None: ub = np.max(_vals)

        if scale=='linear':
            p.set_clim([lb, ub])
        elif scale=='log':
            p.norm = mpl.colors.LogNorm(vmin=lb,vmax=ub)
        elif scale=='symlog':
            p.norm = mpl.colors.SymLogNorm(linthresh=ub/10.,base=10, linscale=0.5, vmin=lb,vmax=ub)
        else:
            raise ValueError('Unrecognized scale parameter')
        
        ax.add_collection(p)

        cbar = plt.colorbar(p, ax=ax, pad=0.01, ticks = [ub,ub/10,lb/10,lb] if scale=='symlog' else None)
        cbar = plot_tools.DraggableColorbar(cbar,p)
        cid = cbar.connect()

        ax.set_title(label)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.axis('scaled')
        
    
    def plot2d_eirene(self, vals,  ax = None, scale='log', label='', lb=None, ub=None, replace_zero=True, **kwargs):
        '''Method to plot 2D fields from EIRENE.

        Parameters
        ----------
        vals : array (self.triangles)
            Data array for an EIRENE variable of interest.
        ax : matplotlib Axes instance
            Axes on which to plot. If left to None, a new figure is created.
        scale : str
            Choice of 'linear','log' and 'symlog' for matplotlib.colors.
        label : str
            Label to set on the colorbar. No label by default.
        lb : float
            Lower bound for colorbar. If left to None, the minimum value in `vals` is used.
        ub : float
            Upper bound for colorbar. If left to None, the maximum value in `vals` is used.
        replace_zero : boolean
            If True (default), replace all zeros in 'vals' with minimum value in 'vals'
        kwargs
            Additional keyword arguments passed to the `tripcolor` function.

        '''
        # Avoid zeros that may derive from low Monte Carlo statistics
        #np.nan_to_num(vals,copy=False)
        
        if replace_zero:
            if np.any(vals==0): vals[vals==0.0] = nsmallest(2,np.unique(vals.flatten()))[1]

        if ax is None:
            fig,ax = plt.subplots(figsize=(8,11))

        if hasattr(self, 'geqdsk') and 'RBBBS' in self.geqdsk:
            # plot LCFS
            ax.plot(self.geqdsk['RBBBS'], self.geqdsk['ZBBBS'], c='k')

        if lb is None:
            lb = np.nanmin(vals)
        if ub is None:
            ub = np.nanmax(vals)
        
        # given quantity is on EIRENE triangulation
        if scale=='linear': norm = None
        elif scale=='log': norm =  mpl.colors.LogNorm(vmin=lb,vmax=ub)
        elif scale=='symlog': norm = mpl.colors.SymLogNorm(linthresh=ub/10.,base=10,
                                        linscale=0.5, vmin=lb,vmax=ub)
        else: raise ValueError('Unrecognized scale parameter')
        
        cntr = ax.tripcolor(self.xnodes, self.ynodes, self.triangles,
                             facecolors=vals.flatten(), norm=norm, **kwargs)

        ax.axis('scaled')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        cbar = plt.colorbar(cntr, ax=ax) # format='%.3g'
        ax.set_title(label)
        cbar = plot_tools.DraggableColorbar(cbar,cntr)
        cid = cbar.connect()


    def get_radial_prof(self, vals, dz_mm=5, theta=0, label='', plot=False):
        '''Extract radial profiles of a quantity "quant" from the SOLPS run. 
        This function returns profiles on the low- (LFS) and high-field-side (HFS) midplane, 
        as well as flux surface averaged (FSA) ones. 

        Parameters
        ----------
        vals : array (self.data('ny'), self.data('nx'))
            Data array for a variable of interest.
        dz_mm : float
            Vertical range [mm] over which quantity should be averaged near the midplane. 
            Mean and standard deviation of profiles on the LFS and HFS will be returned based on
            variations of atomic neutral density within this vertical span.
            Note that this does not apply to the FSA calculation. Default is 5 mm.
        theta : float (0-360)
            Poloidal angle [degrees] at which to take radial profile, measured from 
            0 degrees at Outer Midplane. Default is 0 degrees
        label : string
            Optional string label for plot and legend. Default is empty ('')
        plot : bool
            If True, plot radial profiles. 

        Returns
        -------
        rhop_fsa : 1D array
            Sqrt of poloidal flux grid on which FSA profiles are given.
        prof_fsa : 1D array
            FSA profile on rhop_fsa grid.
        rhop_LFS : 1D array
            Sqrt of poloidal flux grid on which LFS profile  (prof_LFS) is given.
        prof_LFS : 1D array
            Mean LFS midpane profile on rhop_LFS grid.
        prof_LFS_std : 1D array
            Standard deviation of LFS midplane profile on the rhop_LFS grid, based on variations 
            within +/-`dz_mm`/2 millimeters from the midplane. 
        rhop_HFS : 1D array
            Sqrt of poloidal flux grid on which the midplane HFS profile (prof_HFS) is given.
        prof_HFS : 1D array
            Mean HFS midplane profile on rhop_HFS grid.
        prof_HFS_std : 1D array
            Standard deviation of HFS midplane profile on rhop_HFS grid, based on variations 
            within +/-`dz_mm`/2 millimeters from the midplane.
        '''
        
        if self.form=='files' and np.prod(vals.shape) == self.data('crx').shape[1]*self.data('crx').shape[2]:
            # Exclude boundary cells
            vals = vals[1:-1, 1:-1]

        rhop_2D = coords.get_rhop_RZ(self.data('cr'), self.data('cz'), self.geqdsk)
        
        # evaluate FSA radial profile inside the LCFS
        def avg_function(r, z):
            if any(coords.get_rhop_RZ(r, z, self.geqdsk)<np.min(rhop_2D)):
                return np.nan
            else:
                return griddata((self.data('cr').flatten(), self.data('cz').flatten()), vals.flatten(),
                                (r,z), method='linear')

        prof_FSA = self.geqdsk['fluxSurfaces'].surfAvg(function=avg_function)
        rhop_FSA = np.sqrt(self.geqdsk['fluxSurfaces']['geo']['psin'])

        # get R axes on the midplane on the LFS and HFS
        # rule-of-thumb to identify vertical resolution:
        _dz = (np.max(self.data('cz')) - np.min(self.data('cz')))/\
              ((self.data('nx')+self.data('ny'))/10.) 
        mask = (self.data('cz').flatten()>-_dz)&(self.data('cz').flatten()<_dz)
        R_midplane = self.data('cr').flatten()[mask]
        
        R_midplane_lfs = R_midplane[R_midplane>self.geqdsk['RMAXIS']]
        _R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),1000)
        
        R_midplane_hfs = R_midplane[R_midplane<self.geqdsk['RMAXIS']]
        _R_HFS = np.linspace(np.min(R_midplane_hfs), np.max(R_midplane_hfs),1000)

        # get midplane radial profile...
        # ...on the LFS:
        _prof_LFS = griddata((self.data('cr').flatten(),self.data('cz').flatten()), vals.flatten(),
                             (_R_LFS,0.5*dz_mm*1e-3*np.random.random(len(_R_LFS))),
                             #(_R_LFS,np.zeros_like(_R_LFS)),
                             method='linear')
        _prof_LFS[_prof_LFS<0]=np.nan
        R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),100)
        rhop_LFS = coords.get_rhop_RZ(R_LFS,np.zeros_like(R_LFS), self.geqdsk)

        # ... and on the HFS:
        _prof_HFS = griddata((self.data('cr').flatten(),self.data('cz').flatten()), vals.flatten(), #self.quants[quant].flatten(),
                             #(_R_HFS, np.zeros_like(_R_HFS)),
                             (_R_HFS,0.5*dz_mm*1e-3*np.random.random(len(_R_HFS))),
                             method='linear')
        _prof_HFS[_prof_HFS<0]=np.nan
        R_HFS = np.linspace(np.min(R_midplane_hfs), np.max(R_midplane_hfs),100)   
        rhop_HFS = coords.get_rhop_RZ(R_HFS,np.zeros_like(R_HFS), self.geqdsk)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') # we might take the mean of slices with only nan's, but that's OK
            prof_LFS = np.nanmean(_prof_LFS.reshape(-1,10),axis=1) # average across 10 near points
            prof_HFS = np.nanmean(_prof_HFS.reshape(-1,10),axis=1)  # average across 10 near points

            # take std as a measure of variation/noise around chosen location
            prof_LFS_std = np.nanstd(_prof_LFS.reshape(-1,10),axis=1) # std across 10 near points
            prof_HFS_std = np.nanstd(_prof_HFS.reshape(-1,10),axis=1)  # std across 10 near points
            
        # now obtain also the simple poloidal grid slice near the midplane (LFS and HFS)
        # These are commonly used for SOLPS analysis, using the JXA and JXI indices (which we re-compute here)
        Z_core = self.data('cz')[self.unit_r:2*self.unit_r,self.unit_p:3*self.unit_p]
        R_core = self.data('cr')[self.unit_r:2*self.unit_r,self.unit_p:3*self.unit_p]

        # find indices of poloidal grid nearest to Z=0 in the innermost radial shell
        midplane_LFS_idx = np.argmin(np.abs(Z_core[0,R_core[0,:]>self.geqdsk['RMAXIS']]))
        midplane_HFS_idx = np.argmin(np.abs(Z_core[0,R_core[0,:]<self.geqdsk['RMAXIS']]))

        # convert to indices on self.data('cz') and self.data('cr')
        JXI = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]<self.geqdsk['RMAXIS']][midplane_HFS_idx]  # HFS_mid_pol_idx
        JXA = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]>self.geqdsk['RMAXIS']][midplane_LFS_idx] # LFS_mid_pol_idx

        # find rhop along midplane grid chords
        rhop_chord_HFS = coords.get_rhop_RZ(self.data('cr')[:,JXI],self.data('cz')[:,JXI], self.geqdsk)
        rhop_chord_LFS = coords.get_rhop_RZ(self.data('cr')[:,JXA],self.data('cz')[:,JXA], self.geqdsk)

        if plot:
            # compare FSA radial profiles with midplane (LFS and HFS) ones
            fig,ax = plt.subplots()
            ax.plot(rhop_FSA, prof_FSA, label='FSA')
            l = ax.plot(rhop_LFS, prof_LFS, label='LFS midplane')
            ax.errorbar(rhop_LFS, prof_LFS, prof_LFS_std, c=l[0].get_color(), alpha=0.8)
            l = ax.plot(rhop_HFS, prof_HFS, label='HFS midplane')
            ax.errorbar(rhop_HFS, prof_HFS, prof_HFS_std, c=l[0].get_color(), alpha=0.8)
            #ax.plot(rhop_chord_LFS, vals[:,JXA], label='LFS grid midplane')
            #ax.plot(rhop_chord_HFS, vals[:,JXI], label='HFS grid midplane')
            ax.set_xlabel(r'$\rho_p$')
            ax.set_ylabel(label)
            ax.legend(loc='best').set_draggable(True)
            plt.tight_layout()

        return rhop_FSA, prof_FSA, rhop_LFS, prof_LFS, rhop_HFS, prof_HFS


    def get_poloidal_prof(self, vals, plot=False, label='', rhop=1.0, topology='LSN', ax=None):
        '''Extract poloidal profile of a quantity "quant" from the SOLPS run. 
        This function returns a profile of the specified quantity at the designated radial coordinate
        (rhop=1 by default) as a function of the poloidal angle, theta.

        Note that double nulls ('DN') are not yet handled. 

        Parameters
        ----------
        vals : array (self.data('ny'), self.data('nx'))
            Data array for a variable of interest.
        plot : bool
            If True, plot poloidal profile.
        label : string
            Label for plot
        rhop : float
            Radial coordinate, in rho_p, at which to take poloidal surface. Default is 1 (LCFS)            
        ax : matplotlib axes instance
            Axes on which poloidal profile should be plotted. If not given, a new set of axes is created.
            This is useful to possibly overplot profiles at different radii.

        Returns
        -------
        theta_rhop : 1D array
            Poloidal grid measured in degrees from LFS midplane on which prof_rhop is given
        prof_rhop : 1D array
            Mean poloidal profile at rhop on theta_rhop grid.
            
        prof_rhop_std : 1D array
            Standard deviation of poloidal profile at rhop on the theta_rhop grid, based on variations 
            within +/-`dr_mm`/2 millimeters from the surface at rhop. 
        '''
        if self.DN:
            raise ValueError('Detected double-null geometry, not yet handled by this function!')

        if self.form=='files' and np.prod(vals.shape)==self.data('crx').shape[1]*self.data('crx').shape[2]:
            # Exclude boundary cells
            vals = vals[1:-1,1:-1]         

        # find x-point coordinates
        self.find_xpoint()

        _R_points=np.linspace(np.min(self.data('cr')),np.max(self.data('cr')), 202)
        if self.xpoint[1]>0: # USN
            _Z_points = np.linspace(np.min(self.data('cz')), self.xpoint[1], 200)
        else: # LSN (assume not DN)
            _Z_points = np.linspace(self.xpoint[1], np.max(self.data('cz')), 200)

        _R_grid,_Z_grid=np.meshgrid(_R_points,_Z_points,copy=False)
        rhop_2D = coords.get_rhop_RZ(_R_grid,_Z_grid, self.geqdsk)

        # rule-of-thumb to identify radial resolution:
        dr_rhop = (np.max(self.data('cr')) - np.min(self.data('cr')))/(self.data('ny')*10) 
        
        _mask = (rhop_2D<rhop+dr_rhop)&(rhop_2D>rhop-dr_rhop)
        
        _R_rhop = _R_grid[_mask]
        _Z_rhop = _Z_grid[_mask]

        # Need a way to get more resolution of the R and Z coordinates for given rhop
        prof_rhop = griddata((self.data('cr').flatten(), self.data('cz').flatten()),
                             vals.flatten(), (_R_rhop, _Z_rhop), method='cubic')
        
        Rmaxis = self.geqdsk['RMAXIS']
        Zmaxis = self.geqdsk['ZMAXIS']
        
        _LFS_midplane_vect = np.array([np.max(self.data('cr'))-Rmaxis, 0])
        _XP_vect = np.array([self.xpoint[0]-Rmaxis, self.xpoint[1]-Zmaxis])
        _theta_vect = np.array([_R_rhop-Rmaxis, _Z_rhop-Zmaxis]).T
        
        _theta_XP = np.degrees(np.arctan2(np.linalg.det([_LFS_midplane_vect,_XP_vect]),
                                          np.dot(_LFS_midplane_vect,_XP_vect)))

        theta_rhop = [np.degrees(np.arctan2(np.linalg.det([_LFS_midplane_vect,_vect]),
                                            np.dot(_LFS_midplane_vect,_vect))) for _vect in _theta_vect]   

        if _theta_XP < 0:
            theta_rhop = [x+360 if x < _theta_XP else x for x in theta_rhop]
        else:
            theta_rhop = [x+360 if x < -90. else x for x in theta_rhop]
        
        poloidal_prof = np.array(sorted(zip(theta_rhop,prof_rhop))).T
        theta_prof = poloidal_prof[0]
        pol_prof = poloidal_prof[1]

        if plot:

            if ax is None:
                fig,ax = plt.subplots(figsize=(9,5))
                # plot black vertical line at poloidal loc of x-point (left-most in plot)
                ax.axvline(_theta_XP, c='k', label='X point')
                for ang in [-90,0,90,180]:
                    # plot black dashed lines at key angles that help in visualization
                    ax.axvline(x=ang, c='k',ls='--')
                
            ax.semilogy(theta_prof, pol_prof, '.', label=fr'$\rho_p={rhop}$')
            ax.set_xlabel(r'$\theta$ [${}^\circ$]')
            ax.set_ylabel(label)
            ax.legend(loc='best').set_draggable(True)
            plt.tight_layout()

        return poloidal_prof

    def find_gfile(self):
        '''Identify the name of the gEQDSK file from the directory where 
        b2fgmtry is also located. 
        '''
        for filename in os.path.dirname(self.b2fgmtry_path):
            if filename.startswith('g'): # assume only 1 file starts with 'g'
                return filename

    def plot_radial_summary(self, ls='o-b'):
        '''Plot a summary of radial profiles (ne, Te, Ti, nn, nm), 
        at the inner and outer targets, as well as the outer midplane.

        Parameters
        ----------
        ls : str
            
        '''
        out = {}
        for v in ['nx','ny','ns','imp','omp','sep','dsrad','ne','te','ti']:
            out[v] = self.data(v)

        fig, axs= plt.subplots(3,3, figsize=(10,8))
        axs[0,0].set_title('Inner target')
        axs[0,0].plot(out['dsrad'][:,0], out['ne'][:,0], ls)
        axs[0,1].plot(out['dsrad'][:,out['omp'][0]], out['ne'][:,out['omp'][0]], ls)
        axs[0,2].plot(out['dsrad'][:,out['nx'][0]-1], out['ne'][:,out['nx'][0]-1], ls)

        axs[0,1].set_title('Outer midplane')
        axs[1,0].plot(out['dsrad'][:,0], out['te'][:,0], ls)
        axs[1,1].plot(out['dsrad'][:,out['omp'][0]], out['te'][:,0], ls)
        axs[1,2].plot(out['dsrad'][:,out['nx'][0]-1], out['te'][:,0], ls)

        axs[0,2].set_title('Outer target')
        axs[2,0].plot(out['dsrad'][:,0], out['ti'][:,0], ls)
        axs[2,1].plot(out['dsrad'][:,out['omp'][0]], out['ti'][:,0], ls)
        axs[2,2].plot(out['dsrad'][:,out['nx'][0]-1], out['ti'][:,0], ls)
        
        axs[0,0].set_ylabel(r'ne [cm$^{-3}$]')
        axs[1,0].set_ylabel('Ti [eV]')
        axs[2,0].set_ylabel('Te [eV]')
        for ii in [0,1,2]:
            axs[-1,ii].set_xlabel('ds')
            for jj in [0,1,2]:
                axs[ii,jj].grid(True, ls='--')

        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        plt.tight_layout()

    def find_xpoint(self):
        '''Find location of the x-point in (R,Z) coordinates by using the fact
        that it is the only vertex shared by 8 cells. 
        '''
        vertices = {}
        for ny in range(int(self.data('ny'))):
            for nx in range(int(self.data('nx'))):
                for v in range(self.data('crx').shape[0]):
                    vertex = str(self.data('crx')[v,ny,nx]) + ' ' + str(self.data('cry')[v,ny,nx])
                    vertices[vertex] = vertices.get(vertex,0) + 1

        xpoint = list(vertices.keys())[list(vertices.values()).index(8)]
        self.xpoint = np.array([float(val) for val in xpoint.split()])

    def get_3d_path(self, pnt1, pnt2, npt=501, plot=False, ax=None):
        '''Given 2 points in 3D Cartesian coordinates, returns discretized 
        R,Z coordinates along the segment that connects them. 
        
        Parameters
        ----------
        pnt1 : array (3,)
            Cartesian coordinates x,y,z of first path extremum (expected in [m]).
        pnt2 : array (3,)
            Cartesian coordinates x,y,z of second path extremum (expected in [m]).
        npt : int, optional
            Number of points to use for the path discretization.
        plot : bool, optional
            If True, display  displaying result. Default is False. 
        ax : matplotlib axes instance, optional
            If provided, draw 3d path on these axes. Default is to create
            a new figure and also draw the B2.5 grid polygons.

        Returns
        -------
        pathR : array (npt,)
            R [m] points along the path.
        pathZ : array (npt,)
            Z [m] points along the path.
        pathL : array (npt,)
            Length [m] discretization along the path.
        '''
        xyz = np.outer(pnt2 - pnt1, np.linspace(0, 1, int(npt))) + pnt1[:, None]

        # mapping x, y, z to poloidal plane
        pathR, pathZ = np.hypot(xyz[0], xyz[1]), xyz[2]

        pathL = np.linalg.norm(xyz - pnt1[:, None], axis=0)

        if plot:
            if ax is None:
                fig,ax = plt.subplots(1,figsize=(9, 11))
                
                # get polygons describing B2 grid
                p = self.get_b2_patches()
                
                ax.add_collection(p)
                ax.set_xlabel('R [m]')
                ax.set_ylabel('Z [m]')
                ax.axis('scaled')
            
            # now overplot 
            ax.plot(pathR, pathZ, c='b')

        return pathR, pathZ, pathL
        
    def eval_LOS(self, pnt1, pnt2, vals,
                 npt=501, method='linear', plot=False, ax=None, label=None):
        '''Evaluate the SOLPS output `field` along the line-of-sight (LOS)
        given by the segment going from point `pnt1` to point `pnt2` in 3D
        geometry.

        Parameters
        ----------
        pnt1 : array (3,)
            Cartesian coordinates x,y,z of first extremum of LOS.
        pnt2 : array (3,)
            Cartesian coordinates x,y,z of second extremum of LOS.
        vals : array (self.data('ny'), self.data('nx'))
            Data array for a variable of interest.
        npt : int, optional
            Number of points to use for the path discretization.
        method : {'linear','nearest','cubic'}, optional
            Method of interpolation.
        plot : bool, optional
            If True, display variation of the `field` quantity as a function of the LOS 
            path length from point `pnt1`.
        ax : matplotlib axes, optional
            Instance of figure axes to use for plotting. If None, a new figure is created.
        label : str, optional
            Text identifying the LOS under examination.

        Returns
        -------
        array : (npt,)
            Values of requested SOLPS output along the LOS
        '''
        # get R,Z and length discretization along LOS
        pathR, pathZ, pathL = self.get_3d_path(pnt1, pnt2, npt=npt, plot=False)

        # interpolate from SOLPS case (cell centers) onto LOS
        vals_LOS = griddata((self.data('cr').flatten(), self.data('cz').flatten()),
                            vals.flatten(),
                            (pathR, pathZ), method=str(method))

        if plot:
            # plot values interpolated along LOS
            if ax is None:
                fig, ax1 = plt.subplots()
                ax1.set_xlabel('l [m]')
            else:                
                ax1 = ax
            ax1.plot(pathL, vals_LOS, label=label)
            ax1.legend(loc='best').set_draggable(True)
            plt.tight_layout()
            
        return vals_LOS


def apply_mask(triang, geqdsk, max_mask_len=0.4, mask_up=False, mask_down=False):
    '''Function to apply basic masking to a matplolib triangulation. This type of masking
    is useful to avoid having triangulation edges going outside of the true simulation
    grid. 

    Parameters
    ----------
    triang : instance of matplotlib.tri.triangulation.Triangulation
        Matplotlib triangulation object for the (R,Z) grid. 
    geqdsk : dict
        Dictionary containing gEQDSK file values as processed by `omfit_classes.omfit_eqdsk`. 
    max_mask_len : float
        Maximum length [m] of segments within the triangulation. Segments longer
        than this value will not be plotted. This helps avoiding triangulation 
        over regions where no data should be plotted, beyond the actual simulation
        grid.
    mask_up : bool
        If True, values in the upper vertical half of the mesh are masked. 
        Default is False.
    mask_down : bool
        If True, values in the lower vertical half of the mesh are masked. 
        Default is False.
      
    Returns
    -------
    triang : instance of matplotlib.tri.triangulation.Triangulation
        Masked instance of the input matplotlib triangulation object.
    '''
    triangles = triang.triangles
    x = triang.x; y = triang.y
    
    # Find triangles with sides longer than max_mask_len
    xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
    ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    cond_maxd = maxi > max_mask_len

    # second condition: mask upper and/or lower part of the grid
    if mask_up:
        cond_up = np.mean(y[triangles],axis=1)>0.0
    else:
        cond_up = np.mean(y[triangles],axis=1)<1e10  # all True
    if mask_down:
        cond_down = np.mean(y[triangles],axis=1)<=0.0
    else:
        cond_down = np.mean(y[triangles],axis=1)<1e10  # all True

    cond_y = cond_up & cond_down

    # apply mask to points within the LCFS (no triangulation issues here)
    rhop_triangs = coords.get_rhop_RZ(np.mean(x[triangles],axis=1),
                             np.mean(y[triangles],axis=1),
                             geqdsk) 
    center_mask = rhop_triangs < np.min(coords.get_rhop_RZ(x,y,geqdsk))

    # apply masking
    triang.set_mask((cond_maxd & cond_y) | center_mask)

    return triang




def get_fort44_info(NDX,NDY,NATM,NMOL,NION,NSTRA,NCL,NPLS,NSTS,NLIM):
    '''Collection of labels and dimensions for all fort.44 variables, as collected in the 
    SOLPS-ITER 2020 manual.
    '''
    
    fort44_info = {
        'dab2': [r'Atom density ($m^{-3}$)',(NDX,NDY,NATM)],
        'tab2': [r'Atom temperature (eV )', (NDX,NDY,NATM)],
        'dmb2': [r'Molecule density ($m^{-3}$)', (NDX,NDY,NMOL)],
        'tmb2': [r'Molecule temperature (eV )', (NDX,NDY,NMOL)],
        'dib2': [r'Test ion density ($m^{-3}$)', (NDX,NDY,NION)],
        'tib2': [r' Test ion temperature (eV)', (NDX,NDY,NION)],
        'rfluxa': [r'Radial flux density of atoms ($m^{-2} s^{-1}$)', (NDX,NDY,NATM)],
        'rfluxm': [r'Radial flux density of molecules ($m^{-2} s^{-1}$)', (NDX,NDY,NMOL)],
        'pfluxa': [r'Poloidal flux density of atoms ($m^{-2} s^{-1}$)', (NDX,NDY,NATM)],
        'pfluxm': [r'Poloidal flux density of molecules ($m^{-2} s^{-1}$)', (NDX,NDY,NMOL)],
        'refluxa': [r'Radial energy flux density carried by atoms ($W m^{-2}$)', (NDX,NDY,NATM)],
        'refluxm': [r'Radial energy flux density carried by molecules ($W m^{-2}$)', (NDX,NDY,NMOL)],
        'pefluxa': [r'Poloidal energy flux density carried by atoms ($W m^{-2}$)', (NDX,NDY,NATM)],
        'pefluxm': [r'Poloidal energy flux density carried by molecules ($W m^{-2}$)', (NDX,NDY,NMOL)],
        #
        'emiss': [r'$H_\alpha$ emissivity due to atoms ($photons m^{-2} s^{-1}$)', (NDX,NDY)],
        'emissmol': [r'$H_\alpha$ emissivity due to molecules and molecular ions ($photons m^{-2} s^{-1}$)', (NDX,NDY)],
        'srcml': [r'Molecule particle source (A)', (NDX,NDY,NMOL)],
        'edissml': [r'Energy spent for dissociating hydrogenic molecules (W)', (NDX,NDY,NMOL)],
        'wldnek': [r'Heat transferred by neutrals (W), total over strata', (NLIM+NSTS,)],
        'wldnep': [r'Potential energy released by neutrals (W), total over strata', (NLIM+NSTS,)],
        'wldna': [r'Flux of atoms impinging on surface (A), total over strata', (NLIM+NSTS,NATM)],
        'ewlda': [r'Average energy of impinging atoms on surface (eV), total over strata', (NLIM+NSTS,NATM)],
        'wldnm': [r'Flux of molecules impinging on surface (A), total over strata', (NLIM+NSTS,NMOL)],
        'ewldm': [r'Average energy of impinging molecules on surface (eV), total over strata', (NLIM+NSTS,NMOL)],
        'p1,p2': [r'Endpoints of surface (X and Y coordinates, in m), total over strata', (NLIM,)],
        'wldra': [r'Flux of reflected atoms from surface (A), total over strata', (NLIM+NSTS,NATM)],
        'wldrm': [r'Flux of reflected molecules from surface (A), total over strata', (NLIM+NSTS,NMOL)],
    }
    
    for i in np.arange(NSTRA+1):  # from 0 to NSTRA, unlike in manual
        fort44_info.update(
            {f'wldnek({i})': [r'Heat transferred by neutrals (W)', (NLIM+NSTS,)],
             f'wldnep({i})': [r'Potential energy released by neutrals (W)', (NLIM+NSTS,)],
             f'wldna({i})': [r'Flux of atoms impinging on surface (A)', (NLIM+NSTS,NATM)],
             f'ewlda({i})': [r'Average energy of impinging atoms on surface (eV)', (NLIM+NSTS,NATM)],
             f'wldnm({i})': [r'Flux of molecules impinging on surface (A)', (NLIM+NSTS,NMOL)],
             f'ewldm({i})': [r'Average energy of impinging molecules on surface (eV)', (NLIM+NSTS,NMOL)],
             f'wldra({i})': [r'Flux of reflected atoms from surface (A)', (NLIM+NSTS,NATM)],
             f'wldrm({i})': [r'Flux of reflected molecules from surface (A)', (NLIM+NSTS,NMOL)]}
        )


    fort44_info.update(
        {'wldpp': [r'Flux of plasma ions impinging on surface (A), total over strata', (NLIM+NSTS,NPLS)],
         'wldpa': [r'Net flux of atoms emitted from surface (A), total over strata', (NLIM+NSTS,NATM)],
         'wldpm': [r'Net flux of molecules emitted from surface (A), total over strata', (NLIM+NSTS,NMOL)],
         'wldpeb': [r'Power carried by particles emitted from surface (W), total over strata', (NLIM+NSTS,)],
         'wldspt': [r'Flux of sputtered wall material (A), total over strata', (NLIM+NSTS,)],
         'wldspta': [r'Flux of sputtered wall material per atom (A), total over strata', (NLIM+NSTS,NATM)],
         'wldsptm': [r'Flux of sputtered wall material per molecule (A), total over strata', (NLIM+NSTS,NMOL)]}
        )

    for i in np.arange(NSTRA+1): # from 0 to NSTRA, unlike in manual
        fort44_info.update(
            {f'wldpp({i})': [r'Flux of plasma ions impinging on surface (A)', (NLIM+NSTS,NPLS)],
             f'wldpa({i})': [r'Net flux of atoms emitted from surface (A)', (NLIM+NSTS,NATM)],
             f'wldpm({i})': [r'Net flux of molecules emitted from surface (A)', (NLIM+NSTS,NMOL)],
             f'wldpeb({i})': [r'Power carried by particles emitted from surface (W)', (NLIM+NSTS,)],
             f'wldspt({i})': [r'Flux of sputtered wall material (A)', (NLIM+NSTS,)],
             f'wldspta({i})': [r'Flux of sputtered wall material per atom (A)', (NLIM+NSTS,NATM)],
             f'wldsptm({i})': [r'Flux of sputtered wall material per molecule (A)', (NLIM+NSTS,NMOL)],
             }
        )

    fort44_info.update(
        {
            
            'isrftype': [r'ILIIN surface type variable in Eirene', (NLIM+NSTS,)],
            'wlarea': [r'Surface area (m2)', (NLIM+NSTS,)],
            'wlabsrp(A)': [r'Absorption rate for atoms', (NATM, NLIM+NSTS)],
            'wlabsrp(M)': [r'Absorption rate for molecules', (NMOL, NLIM+NSTS)],
            'wlabsrp(I)': [r'Absorption rate for test ions', (NION, NLIM+NSTS)],
            'wlabsrp(P)': [r'Absorption rate for plasma ions', (NPLS, NLIM+NSTS)],
            'wlpump(A)': [r'Pumped flux per atom (A)', (NATM, NLIM+NSTS)],
            'wlpump(M)': [r'Pumped flux per molecule (A)', (NMOL, NLIM+NSTS)],
            'wlpump(I)': [r'Pumped flux per test ion (A)', (NION, NLIM+NSTS)],
            'wlpump(P)': [r'Pumped flux per plasma ion (A)', (NPLS, NLIM+NSTS)],
            'eneutrad': [r'Radiation rate due to atoms (W)', (NDX,NDY,NATM)],
            'emolrad': [r'Radiation rate due to molecules (W)', (NDX,NDY,NMOL)],
            'eionrad': [r'Radiation rate due to test ions (W)', (NDX,NDY,NION)],
            # eirdiag rather than eirdiag_nds, as in manual...
            'eirdiag': [r'Indices for segments on resolved non-standard surfaces', (5*NSTS +1,)],
            'sarea_res': [r'Surface area of surface segment (m2)', (NCL,)],
            'wldna_res': [r'Flux of atoms impinging on surface segment (A)', (NATM, NCL)],
            'wldnm_res': [r'Flux of molecules impinging on surface segment (A)', (NMOL, NCL)],
            'ewlda_res': [r'Average energy of impinging atoms on surface segment (eV)', (NATM, NCL)],
            'ewldm_res': [r'Average energy of impinging molecules on surface segment (eV)', (NMOL, NCL)],
            'ewldea_res': [r'Energy flux carried by emitted atoms from surface segment (W)', (NATM, NCL)],
            'ewldem_res': [r'Energy flux carried by emitted molecules from surface segment (W)', (NMOL, NCL)],
            'ewldrp_res': [r'Total energy flux carried by emitted particles from surface segment (W)', (NCL,)],
            'ewldmr_res': [r'Flux of emitted molecules from recycling atoms (A)', (NMOL, NCL)],
            'wldspt_res': [r'Flux of sputtered wall material (A)', (NCL,)],
            'wldspta_res': [r'Flux of sputtered wall material per atom (A)', (NCL, NATM)],
            'wldsptm_res': [r'Flux of sputtered wall material per molecule (A)', (NCL, NMOL)],
            'wlpump_res(A)': [r'Pumped flux per atom (A)', (NCL, NATM)],
            'wlpump_res(M)': [r'Pumped flux per molecule (A)', (NCL, NMOL)],
            'wlpump_res(I)': [r'Pumped flux per test ion (A)', (NCL, NION)],
            'wlpump_res(P)': [r'Pumped flux per plasma ion (A)', (NCL, NPLS)],
            'ewldt_res': [r'Total wall power loading from Eirene particles', (NCL,)],
            'pdena_int': [r'Integral number of atoms over the entire Eirene computational grid', (NATM, NSTRA+1)],
            'pdenm_int': [r'Integral number of molecules over the entire Eirene computational grid', (NMOL, NSTRA+1)],
            'pdeni_int': [r'Integral number of test ions over the entire Eirene computational grid', (NION, NSTRA+1)],
            'pdena_int_b2': [r'Integral number of atoms over the B2.5 computational grid', (NATM, NSTRA+1)],
            'pdenm_int_b2': [r'Integral number of molecules over the B2.5 computational grid', (NMOL, NSTRA+1)],

            'pdeni_int_b2': [r'Integral number of test ions over the B2.5 computational grid', (NION, NSTRA+1)],
            'edena_int': [r'Integral energy carried by atoms over the entire Eirene computational grid (J)', (NATM,NSTRA+1)],
            'edenm_int': [r'Integral energy carried by molecules over the entire Eirene computational grid (J)', (NMOL,NSTRA+1)],
            'edeni_int': [r'Integral energy carried by test ions over the entire Eirene computational grid (J)', (NION,NSTRA+1)],
            'edena_int_b2': [r'Integral energy carried by atoms over the B2.5 computational grid (J)', (NATM, NSTRA+1)],
            'edenm_int_b2': [r'Integral energy carried by molecules over the B2.5 computational grid (J)', (NMOL, NSTRA+1)],
            'edeni_int_b2': [r'Integral energy carried by test ions over the B2.5 computational grid (J)', (NION, NSTRA+1)]
        }
    )

    # extra, undocumented
    fort44_info.update({'wall_geometry' : [r'Wall geometry points', (4*NLIM,)]})
    return fort44_info

def get_fort46_info(NTRII,NATM,NMOL,NION):
    '''Collection of labels and dimensions for all fort.46 variables, as collected in the 
    SOLPS-ITER 2020 manual.
    '''
    
    fort46_info = {
        'PDENA': [r'Atom particle density ($cm^{-3}$)',(NTRII,NATM)],
        'PDENM': [r'Molecule particle density ($cm^{-3}$)', (NTRII,NMOL)],
        'PDENI': [r'Test ion particle density ($cm^{-3}$)', (NTRII,NION)],
        'EDENA': [r'Energy density carried by atoms ($eV*cm^{-3}$)', (NTRII,NATM)],
        'EDENM': [r'Energy density carried by molecules ($eV*cm^{-3}$)', (NTRII,NMOL)],
        'EDENI': [r'Energy density carried by test ions ($eV*cm^{-3}$)', (NTRII,NION)],
        'VXDENA': [r'X-directed momentum density carried by atoms ($g*cm^{-2}*s^{-1}$)', (NTRII,NATM)],
        'VXDENM': [r'X-directed momentum density carried by molecules ($g*cm^{-2}*s^{-1}$)', (NTRII,NMOL)],
        'VXDENI': [r'X-directed momentum density carried by test ions ($g*cm^{-2}*s^{-1}$)', (NTRII,NION)],
        'VYDENA': [r'Y -directed momentum density carried by atoms ($g*cm^{-2}*s^{-1}$)', (NTRII,NATM)],
        'VYDENM': [r'Y -directed momentum density carried by molecules ($g*cm^{-2}*s^{-1}$)', (NTRII,NMOL)],
        'VYDENI': [r'Y -directed momentum density carried by test ions ($g*cm^{-2}*s^{-1}$)', (NTRII,NION)],
        'VZDENA': [r'Z-directed momentum density carried by atoms ($g*cm^{-2}*s^{-1}$)', (NTRII,NATM)],
        'VZDENM': [r'Z-directed momentum density carried by molecules ($g*cm^{-2}*s^{-1}$)', (NTRII,NMOL)],
        'VZDENI': [r'Z-directed momentum density carried by test ions ($g*cm^{-2}*s^{-1}$)', (NTRII,NION)],
        'VOLUMES': [r'Triangle volumes ($cm^{3}$)', (NTRII)],
        'PUX': [r'X-component of the poloidal unit vector at the triangle center', (NTRII)],
        'PUY': [r'Y-component of the poloidal unit vector at the triangle center', (NTRII)],
        'PVX': [r'X-component of the radial unit vector at the triangle center', (NTRII)],
        'PVY': [r'Y-component of the radial unit vector at the triangle center', (NTRII)],
    }
    
    return fort46_info



def get_mdsmap():
    """Load dictionary allowing a mapping of chosen variables with 
    SOLPS variable names on MDS+.
    """
    ident = '\IDENT::TOP:'
    identimp = '\IDENT::TOP.IMPSEP'
    identomp = '\IDENT::TOP.OMPSEP'

    snapshot_root = '\SOLPS::TOP.SNAPSHOT:'
    snaptopdims = '\SNAPSHOT::TOP.DIMENSIONS:'
    snaptopgrid = '\SNAPSHOT::TOP.GRID:'
    snaptop = '\SNAPSHOT::TOP:'

    timedept1 = '\TIMEDEP::TOP.TARGET1:'
    timedept2 = '\TIMEDEP::TOP.TARGET2:'
    timedept3 = '\TIMEDEP::TOP.TARGET3:'
    timedept4 = '\TIMEDEP::TOP.TARGET4:'

    timedepomp = '\TIMEDEP::TOP.OMP:'
    timedepimp = '\TIMEDEP::TOP.IMP:'

    return {
        # general
        'user': ident+'USER',
        'version': ident+'VERSION',
        'solpsversion': ident+'SOLPSVERSION',
        'directory': ident+'directory',
        'exp': ident+'EXP',
        'time': ident+'TIME', # Experiment time point derived from the b2md_namelist in b2md.dat
        'shot': ident+'SHOT', # experimental shot number that was simulated
        
        # Dimensions
        'nx': snaptopdims+'NX', # grid size in pol direction
        'ny': snaptopdims+'NY', # grid size in rad direction
        'ns': snaptopdims+'NS', # number of species
        'natm': snaptopdims+'NATM',  # number of atomic species
        'nion': snaptopdims+'NION',  # number of ion species
        'nmol': snaptopdims+'NMOL',  # number of molecular species

        # Indices
        'imp': snaptopdims+'IMP',  # index of inner midplane
        'omp': snaptopdims+'OMP',  # index of outer midplane
        'sep': snaptopdims+'SEP', # position of the separatrix
        'targ1': snaptopdims+'TARG1',  # position of target 1 (usually inner)
        'targ2': snaptopdims+'TARG2', # position of target 2
        'targ3': snaptopdims+'TARG3', # position of target 3
        'targ4': snaptopdims+'TARG4', # position of target 4 (usually outer)
        'regflx': snaptopgrid+'REGFLX', # x-directed flux region indices
        'regfly': snaptopgrid+'REGFLY', # y-directed flux region indices
        'regvol': snaptopgrid+'REGVOL',  # volume region indices

        # Neighbours
        'bottomix': snaptopgrid+'BOTTOMIX',  # bottom ix neighbourhood array
        'bottomiy': snaptopgrid+'BOTTOMIY',  # bottom iy neighbourhood array
        'leftix': snaptopgrid+'LEFTIX',   # left ix neighbourhood array
        'leftiy': snaptopgrid+'LEFTIY',  # left iy neighbourhood array
        'rightix': snaptopgrid+'RIGHTIX',  # right ix neighbourhood array
        'rightiy': snaptopgrid+'RIGHTIY' , # right iy neighbourhood array
        'topix': snaptopgrid+'TOPIX',  # top ix neighbourhood array
        'topiy': snaptopgrid+'TOPIY',  # top iy neighbourhood array

        # Grid and vessel geometry
        'crx': snaptopgrid+'R',  # R coordinates of cell vertices
        'cry': snaptopgrid+'Z',  # Z coordinates of cell vertices
        'cr': snaptopgrid+'CR',    # R coordinate of the cell center
        'cr_x': snaptopgrid+'CR_X', # R-coordinate of the cell x-face
        'cr_y': snaptopgrid+'CR_Y', # R-coordinate of the cell y-face
        'cz': snaptopgrid+'CZ',  # Z coordinate of the cell center
        'cz_x': snaptopgrid+'CZ_X', # Z-coordinate of the cell x-face
        'cz_y': snaptopgrid+'CZ_Y',  # Z-coordinate of the cell y-face
        'hx': snaptop+'HX',   # length of cell
        'hy': snaptop+'HY',   # width of cell
        'hy1': snaptop+'HY1',  # corrected width of cell
        'dspar': snaptopgrid+'DSPAR',  # parallel distance
        'dspol': snaptopgrid+'DSPOL',  # poloidal distance
        'dsrad': snaptopgrid+'DSRAD', # radial distance
        'sx': snaptop+'SX',   # poloidal contact area
        'sxperp': snaptop+'SXPERP',  # poloidal area perpendicular to flux tube
        'sy': snaptop+'SY',   # ?
        'vessel': snaptopgrid+'VESSEL',   # vessel structure

        # Plasma characterization
        'am': snaptopgrid+'AM',  # atom/ion mass
        'za': snaptopgrid+'ZA',  # atom charge
        'zn': snaptopgrid+'ZN',  # nuclear charge
        'bb': snaptop+'B',  # B field
        'po': snaptop+'PO', # electric potential
        'pot': snaptopgrid+'POT',  # potential energy
        'poti': snaptopgrid+'POTI', # ionization potential
        'qz': snaptop+'QZ',  #?
        'visc': snaptop+'VISC',  # viscosity
        'vol': snaptop+'VOL',  # volume

        # Densities
        'ne': snaptop+'NE',
        'na': snaptop+'NA', # ion/neutral density
        'dab2': snaptop+'DAB2',  # atom density
        'dmb2': snaptop+'DMB2',  # molecular density
        'dib2': snaptop+'DIB2', # ion density

        # Temperatures
        'te': snaptop+'TE',
        'ti': snaptop+'TI',
        'tab2': snaptop+'TAB2', # neutral atom temperature
        'tmb2': snaptop+'TMB2', # neutral molecular temperature
        'tib2': snaptop+'TIB2',  # test ion temperature

        # Velocities
        'ua': snaptop+'UA',     # ion/neutral parallel velocity
        'vlax': snaptop+'VLAX', # poloidal pinch velocity
        'vlay': snaptop+'VLAY', # radial pinch velocity

        # Fluxes
        'fchx': snaptop+'FCHX',  # poloidal current
        'fchx_32': snaptop+'FCHX_32',  # convective poloidal current
        'fchx_52': snaptop+'FCHX_52',  # conductive poloidal current        
        'fchy': snaptop+'FCHY',  # radial current
        'fchy_32': snaptop+'FCHY_32',  # convective radial current
        'fchy_52': snaptop+'FCHY_52',  # conductive radial current        
        'fhex': snaptop+'FHEX', # poloidal electron energy flux
        'fhey': snaptop+'FHEY', # radial electron energy flux
        'fhix': snaptop+'FHIX', # poloidal ion energy flux
        'fhiy': snaptop+'FHIY', # radial ion energy flux
        'fhjx': snaptop+'FHJX', # poloidal electrostatic energy flux
        'fhjy': snaptop+'FHJY', # radial electrostatic energy flux
        'fhmx': snaptop+'FHMX', # poloidal kinetic energy flux
        'fhmy': snaptop+'FHMY', # radial kinetic energy flux
        'fhpx': snaptop+'FHPX', # poloidal potential energy flux
        'fhpy': snaptop+'FHPY', # radial potential energy flux
        'fhtx': snaptop+'FHTX', # poloidal total energy flux
        'fhty': snaptop+'FHTY', # radial total energy flux
        'fmox': snaptop+'FMOX', # poloidal momentum flux
        'fmoy': snaptop+'FMOY', # radial momentum flux
        'fnax': snaptop+'FNAX', # poloidal particle flux
        'fnax_32': snaptop+'FNAX_32', # 3/2 piece
        'fnax_52': snaptop+'FNAX_52', # 5/2 piece
        'fnay': snaptop+'FNAY', # radial particle flux
        'fnay_32': snaptop+'FNAY_32', # 3/2 piece
        'fnay_52': snaptop+'FNAY_52', # 5/2 piece
        'pefa': snaptop+'PEFA',   # poloidal atomic energy flux
        'pefm': snaptop+'PEFM',   # poloidal molecular energy flux
        'pfla': snaptop+'PFLA',  # poloidal atomic flux
        'pflm': snaptop+'PFLM',  # poloidal molecular flux
        'refa': snaptop+'REFA',  # radial atomic energy flux
        'refm': snaptop+'REFM',  # radial molecular energy flux
        'rfla': snaptop+'RFLA', # radial atomic flux (?)
        'rflm': snaptop+'RFLM', # radial molecular flux (?)

        # Coefficients
        'alf': snaptop+'ALF',  # thermoelectric coefficient
        'dna0': snaptop+'D',  # D
        'dp': snaptop+'DP',   # Dpa
        'kye': snaptop+'KYE', # kye
        'kyi': snaptop+'KYI', # kyi
        'kyi0': snaptop+'KYI0', # kyi0
        'sig': snaptop+'SIG', # anomalous conductivity
        'rpt': snaptop+'RPT', # Cumulative Ionisation Potential (function of ix,iy,is)

        # Rates, losses and sources
        'rcxhi': snaptop+'RCXHI',  # CX ion energy neutral losses
        'rcxmo': snaptop+'RCXMO',  # CX momentum neutral losses
        'rcxna': snaptop+'RCXNA',  # CX particle neutral losses
        'rqahe': snaptop+'RQAHE',  # Electron cooling rate
        'rqbrm': snaptop+'RQBRM',  # Bremsstrahlung radiation rate
        'rqrad': snaptop+'RQRAD',  # Line radiation rate
        'rrahi': snaptop+'RRAHI',  # Recombination ion energy losses
        'rramo': snaptop+'RRAMO',  # Recombination momentum losses
        'rrana': snaptop+'RRANA',  # Recombination particle losses
        'rsahi': snaptop+'RSAHI',  # Ionization ion energy losses
        'rsamo': snaptop+'RSAMO',  # Ionization momentum losses
        'rsana': snaptop+'RSANA',  # Ionization particle losses

        'smo': snaptop+'SMO',           # Parallel momentum source
        'smq': snaptop+'SMQ',           # momentum sources from atomic physics
        'b2npmo_smav': snaptop+'SMAV',  # additional viscosity term
        'resmo': snaptop+'RESMO',       # Momentum equation residual

        # Texts
        'textan': snaptop+'TEXTAN',      # atom species
        'textcomp': snaptop+'TEXTCOMP',  # components
        'textin': snaptop+'TEXTIN',      # test ion species
        'textmn': snaptop+'TEXTMN',      # molecular species
        'textpl': snaptop+'TEXTPL',      # plasma species

        # Target 1 -- usually inner
        'TARGET1_na': timedept1+'AN',  # atom density
        'TARGET1_da': timedept1+'DA',  # area element
        'TARGET1_ds': timedept1+'DS',  # S-S_sep
        'TARGET1_fc': timedept1+'FC',  # poloidal current
        'TARGET1_fe': timedept1+'FE',  # poloidal electron energy flux
        'TARGET1_fi': timedept1+'FI',  # poloidal ion energy flux
        'TARGET1_fl': timedept1+'FL',  # poloidal electron flux
        'TARGET1_fn': timedept1+'FN',  # poloidal spec one flux
        'TARGET1_fo': timedept1+'FO',  # poloidal ion flux
        'TARGET1_ft': timedept1+'FT',  # poloidal total energy flux
        'TARGET1_mn': timedept1+'MN',  # molecule density
        'TARGET1_ne': timedept1+'NE',  # electron density
        'TARGET1_po': timedept1+'PO',  # potential
        'TARGET1_te': timedept1+'TE',  # electron temperature
        'TARGET1_ti': timedept1+'TI',  # ion temperature
        'TARGET1_tp': timedept1+'TP',  # target surface temperature

        'TARGET2_na': timedept2+'AN',  # atom density
        'TARGET2_da': timedept2+'DA',  # area element
        'TARGET2_ds': timedept2+'DS',  # S-S_sep
        'TARGET2_fc': timedept2+'FC',  # poloidal current
        'TARGET2_fe': timedept2+'FE',  # poloidal electron energy flux
        'TARGET2_fi': timedept2+'FI',  # poloidal ion energy flux
        'TARGET2_fl': timedept2+'FL',  # poloidal electron flux
        'TARGET2_fn': timedept2+'FN',  # poloidal spec one flux
        'TARGET2_fo': timedept2+'FO',  # poloidal ion flux
        'TARGET2_ft': timedept2+'FT',  # poloidal total energy flux
        'TARGET2_mn': timedept2+'MN',  # molecule density
        'TARGET2_ne': timedept2+'NE',  # electron density
        'TARGET2_po': timedept2+'PO',  # potential
        'TARGET2_te': timedept2+'TE',  # electron temperature
        'TARGET2_ti': timedept2+'TI',  # ion temperature
        'TARGET2_tp': timedept2+'TP',  # target surface temperature

        'TARGET3_na': timedept3+'AN',  # atom density
        'TARGET3_da': timedept3+'DA',  # area element
        'TARGET3_ds': timedept3+'DS',  # S-S_sep
        'TARGET3_fc': timedept3+'FC',  # poloidal current
        'TARGET3_fe': timedept3+'FE',  # poloidal electron energy flux
        'TARGET3_fi': timedept3+'FI',  # poloidal ion energy flux
        'TARGET3_fl': timedept3+'FL',  # poloidal electron flux
        'TARGET3_fn': timedept3+'FN',  # poloidal spec one flux
        'TARGET3_fo': timedept3+'FO',  # poloidal ion flux
        'TARGET3_ft': timedept3+'FT',  # poloidal total energy flux
        'TARGET3_mn': timedept3+'MN',  # molecule density
        'TARGET3_ne': timedept3+'NE',  # electron density
        'TARGET3_po': timedept3+'PO',  # potential
        'TARGET3_te': timedept3+'TE',  # electron temperature
        'TARGET3_ti': timedept3+'TI',  # ion temperature
        'TARGET3_tp': timedept3+'TP',  # target surface temperature

        # target 4 - usually outer target
        'TARGET4_na': timedept4+'AN',  # atom density
        'TARGET4_da': timedept4+'DA',  # area element
        'TARGET4_ds': timedept4+'DS',  # S-S_sep
        'TARGET4_fc': timedept4+'FC',  # poloidal current
        'TARGET4_fe': timedept4+'FE',  # poloidal electron energy flux
        'TARGET4_fi': timedept4+'FI',  # poloidal ion energy flux
        'TARGET4_fl': timedept4+'FL',  # poloidal electron flux
        'TARGET4_fn': timedept4+'FN',  # poloidal spec one flux
        'TARGET4_fo': timedept4+'FO',  # poloidal ion flux
        'TARGET4_ft': timedept4+'FT',  # poloidal total energy flux
        'TARGET4_mn': timedept4+'MN',  # molecule density
        'TARGET4_ne': timedept4+'NE',  # electron density
        'TARGET4_po': timedept4+'PO',  # potential
        'TARGET4_te': timedept4+'TE',  # electron temperature
        'TARGET4_ti': timedept4+'TI',  # ion temperature
        'TARGET4_tp': timedept4+'TP',  # target surface temperature

        # Midplanes
        'OMP_ds': timedepomp+'DS',    
        'OMP_te': timedepomp+'TE',
        'OMP_ti': timedepomp+'TI',
        'OMP_ne': timedepomp+'NE',

        'IMP_ds': timedepimp+'DS',
        'IMP_te': timedepimp+'TE',
        'IMP_ti': timedepimp+'TI',
        'IMP_ne': timedepimp+'NE',
    }

