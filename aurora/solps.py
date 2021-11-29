'''Aurora tools to read SOLPS results and extract atomic neutral density from EIRENE output.
These enable examination of charge exchange recombination and radiation due to the interaction 
of heavy ions and thermal neutrals.
'''
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
    '''Read SOLPS output and prepare for Aurora impurity-neutral analysis. 
    
    Parameters
    ----------
    form : str
        Form of SOLPS output to be loaded, one of {'files','mdsplus'}. 
        If `form="files"`, SOLPS results are in files named 'b2fstate', 'b2fgmtry', etc.
        If `form="mdsplus"`, the SOLPS output is fetched from a MDS+ server.
    solps_id : str or int
        Identifier of SOLPS run to load. If `form="files"`, this is expected to be a str
        indicaticating the directory name where results are stored. 
        If `form="mdsplus"`, this should be an integer identifying the SOLPS run on 
        the provided MDS+ server.

    Several keyword arguments are needed, depending on the `form` input.

    Keyword arguments
    -----------------
    path : str (needed if `form=="files"`)
        Path to SOLPS output files. This path indicates where to find the directory 
        named "baserun" and the `solps_id` directory provided as input.
    geqdsk : str (needed if `form=="files"`)
        Path to the geqdsk to load from disk.
        Users may also directly provide an instance of the 
        `omfit_classes.omfit_geqdsk.OMFITgeqdsk` 
        class that already contains the processed gEQDSK file. 
    server : str (needed if `form=="mdsplus"`)
        MDS+ server to load SOLPS results. Default is 'solps-mdsplus.aug.ipp.mpg.de:8001'.
    tree : str (needed if `form=="mdsplus"`)
        Name of MDS+ tree to load data from.    

    '''
    def __init__(self, form, solps_id, **kwargs):
        self.form = str(form)
        self.solps_id = solps_id
        
        # ensure one of 3 forms of input
        if self.form not in ['files', 'mdsplus']:
            raise ValueError('Unrecognized form of SOLPS output')

        if self.form == 'files' and (not isinstance(self.solps_id, str)):
            raise ValueError('A directory name is required as a string for form="files"')
        if self.form == 'mdsplus' and (not isinstance(self.solps_id, int)):
            raise ValueError('An integer number is needed as "solps_id" when `form="mdsplus"`')
        
        if self.form=='files':
            if not 'path' in kwargs:
                raise ValueError('A path to the SOLPS output files is required \n'+\
                                 'for input `form="files"`')
            self.path = kwargs['path']

            # load omfit_classes here to avoid needing it upon aurora import
            from omfit_classes import omfit_eqdsk
            
            if 'geqdsk' in kwargs:
                if isinstance(kwargs['geqdsk'], str):
                    # load geqdsk file
                    self.geqdsk = omfit_eqdsk.OMFITgeqdsk(geqdsk)
                else:
                    # assume geqdsk was already loaded via OMFITgeqdsk
                    self.geqdsk = kwargs['geqdsk']
            else:
                # try to find gEQDSK and load it
                _gfile_name = self.find_gfile()
                _gfile_path = self.path+os.sep+'baserun'+os.sep+_gfile_name
                if _gfile_name is None:
                    raise ValueError(f'A geqdsk is required for SOLPS outform form "{self.form}"')
                self.geqdsk = omfit_eqdsk.OMFITgeqdsk(_gfile_path)
                

        if self.form=='files':
            
            from omfit_classes import omfit_solps
            self.b2fstate = omfit_solps.OMFITsolps(self.path+os.sep+self.solps_id+os.sep+'b2fstate')
            self.geom = omfit_solps.OMFITsolps(self.path+os.sep+'baserun'+os.sep+'b2fgmtry')

            self.nx,self.ny = self.geom['nx,ny']
            
            # (0,:,:): lower left corner, (1,:,:): lower right corner
            # (2,:,:): upper left corner, (3,:,:): upper right corner.
            
            self.crx = self.geom['crx'].reshape(4,self.ny+2,self.nx+2)  # horizontal 
            self.cry = self.geom['cry'].reshape(4,self.ny+2,self.nx+2)  # vertical

            # figure out if single or double null
            self.double_null = self.cry.shape[2]%4==0
            
            # uncertain units for splitting of radial and poloidal regions...
            self.unit_p = (self.crx.shape[2]-4 if self.double_null else self.crx.shape[2])//4
            self.unit_r = (self.crx.shape[1]-2)//2

            if self.double_null:
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

            # now, load data arrays
            self.load_data() #P_idxs=self.P_idxs, R_idxs=self.R_idxs)  # TODO: enable subselection of regions
        
        elif form=='mdsplus':
            if (not 'server' in kwargs) or (not 'tree' in kwargs):
                raise ValueError('Both MDS+ server and tree are required')
            self.server = kwargs['server']
            self.tree = kwargs['tree']
            
            # load variable name map to MDS+ tree
            self._mdsmap()

            self.quants = {'ne': self._getvar('ne'),
                           'Te': self._getvar('te'),
                           'nn': self._getvar('ti'),
                           'Tn': self._getvar('dab2'),
                           'nm': self._getvar('tab2'),
                           'Tm': self._getvar('dmb2'),
                           'Ti': self._getvar('tmb2'),
            }


            # loading of entire tree:
            #mds_data = omfit_mds.OMFITmds(self.server, self.tree, self.solps_id)
            
            #grid = snapshot['GRID']
            #self.crx = grid['CR_X']
            #self.cry = grid['CR_Y']

            
        # set zero densities equal to min to avoid log issues
        #self.quants['nn'][self.quants['nn']==0.0] = nsmallest(2,np.unique(self.quants['nn'].flatten()))[1]

    def _getvar(self, varname):
        '''Fetch data either from files or MDS+ tree.'''
        if self.form=='mdsplus':
            from omfit_classes import omfit_mds
            # load from MDS+, mapping variable name to tree path
            return omfit_mds.OMFITmdsValue(self.server, self.tree, self.solps_id,
                                      TDI = self.mdsmap[varname]).data()
        else:
            if hasattr(self, varname):
                return getattr(self, varname)
            else:
                raise ValueError(f'Could not fetch variable {varname} from {self.form} files')

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
        
        Returns
        ------------
        quants : dict
            Dictionary containing 'R','Z' coordinates for 2D maps of each field requested by user.
        '''
        if P_idxs is None:
            if Pmax is None: Pmax = self.nx
            if Pmin is None: Pmin = 0
            P_idxs = np.arange(Pmin,Pmax)
        else:
            pass # user provided list of poloidal grid indices

        if R_idxs is None:
            if Rmax is None: Rmax = self.ny
            if Rmin is None: Rmin = 0
            R_idxs = np.arange(Rmin,Rmax)
        else:
            pass # user provided list of radial grid indices

        if fields is None:
            fields = ['ne','Te','nn','Tn','nm','Tm','Ti']
            
        self.quants = quants = {}

        # eliminate end (buffer) points of grid
        self.R = np.mean(self.crx,axis=0)[1:-1,1:-1][R_idxs,:][:,P_idxs]
        self.Z = np.mean(self.cry,axis=0)[1:-1,1:-1][R_idxs,:][:,P_idxs]

        self.fort44 = self.load_fort44()
        self.fort46 = self.load_fort46()

        quants['ne'] = self.b2fstate['ne'][1:-1,1:-1][R_idxs,:][:,P_idxs] # m^-3
        quants['Te'] = self.b2fstate['te'][1:-1,1:-1][R_idxs,:][:,P_idxs]/constants.e # eV
        quants['Ti'] = self.b2fstate['ti'][1:-1,1:-1][R_idxs,:][:,P_idxs]/constants.e # eV

        # density of atomic species if B2 fluid neutral model is used:
        #quants['nn'] = self.b2fstate['na'][0,R_idxs,:][:,P_idxs] # m^-3   # only neutral component

        # assume that EIRENE neutral model is used:
        nn44 = self.fort44['dab2'][:,:,0].T
        nn44[nn44==0.0] = nsmallest(2,np.unique(nn44.flatten()))[1]
        quants['nn'] = nn44[R_idxs,:][:,P_idxs] # m^-3   # only neutral component
        quants['Tn'] = self.fort44['tab2'][:,:,0].T[R_idxs,:][:,P_idxs]/constants.e # eV

        quants['nm'] = self.fort44['dmb2'][:,:,0].T[R_idxs,:][:,P_idxs] # D molecular density
        quants['Tm'] = self.fort44['tmb2'][:,:,0].T[R_idxs,:][:,P_idxs] # D molecular temperature

        # EIRENE nodes and triangulation
        self.xnodes, self.ynodes, self.triangles = self.load_eirene_mesh()

        # identify species (both B2 and EIRENE):
        self.species_id()

            
            
    def species_id(self):
        '''Identify species included in SOLPS run, both for B2 and EIRENE quantities.
        This is only designed to work in the "full" data format.
        '''
        #b2_Zns = self.b2fstate['zn']  # nuclear charge (not ionization state)
        b2_Ans = self.b2fstate['am'] # nuclear masses (2 for H/D/T, 4 for He, etc.)

        # TODO: separate use of zamin and zamax for bundling
        b2_Zas = self.b2fstate['zamin'] # atom charge (min over bundle)
        #b2_Za_max = self.b2fstate['zamax'] # atom charge (max over bundle)
        
        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        self.b2_species = b2_species = {}
        for ii,(Zi,An) in enumerate(zip(b2_Zas,b2_Ans)):
            # get name of atomic species from charge and atomic mass number
            out = atomic_element(Z_ion=Zi, A=An)
            _spec = list(out.keys())[0]
            cs_str = f'{int(Zi)}+' if int(Zi)!=0 else '0'
            b2_species[ii] = {'symbol': out[_spec]['symbol']+cs_str, 'Z':int(Zi),'A':int(An)}

        self.eirene_species = eirene_species = {'atm':{}, 'mol':{}, 'ion':{}}
        _atm = [spec for spec in self.fort44['species'] if ((not any(map(str.isdigit, spec)))&('+' not in spec)) ]
        for ss,atm in enumerate(_atm): eirene_species['atm'][ss] = atm
        _mol = [spec for spec in self.fort44['species'] if ((any(map(str.isdigit, spec)))&('+' not in spec))]
        for ss,mol in enumerate(_mol): eirene_species['mol'][ss] = mol
        _ion = [spec for spec in self.fort44['species'] if '+'  in spec]
        for ss,ion in enumerate(_ion): eirene_species['ion'][ss] = ion


    def load_mesh_extra(self):
        '''Load the mesh.extra file.
        '''
        with open(self.path+os.sep+'baserun'+os.sep+'mesh.extra', 'r') as f:
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
        with open(self.path +os.sep+self.solps_id+os.sep+'fort.44', 'r') as f:
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
        with open(self.path +os.sep+self.solps_id+os.sep+'fort.46', 'r') as f:
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
        nodes=np.fromfile(self.path+os.sep+'baserun'+os.sep+'fort.33',sep=' ')
        n=int(nodes[0])
        xnodes=nodes[1:n+1]/100  # cm -->m
        ynodes=nodes[n+1:]/100
        
        # EIRENE triangulation
        triangles = np.loadtxt(self.path+os.sep+'baserun'+os.sep+'fort.34',
                               skiprows=1, usecols=(1,2,3))

        return xnodes, ynodes, triangles-1  # -1 for python indexing

    def plot_wall_geometry(self):
        '''Method to plot vessel wall segment geometry from wall_geometry field in fort.44 file'''
        
        out=self.load_fort44()
        wall_geometry=out['wall_geometry']
        
        Wall_Seg=[]
        RR=wall_geometry[0::2]
        ZZ=wall_geometry[1::2]
        NLIM=out['NLIM']
        
        for i in range(0,NLIM):
            line=[(RR[2*i],ZZ[2*i]),(RR[2*i+1],ZZ[2*i+1])]
            Wall_Seg.append(line)
            
        Wall_Collection=mc.LineCollection(Wall_Seg,colors='b',linewidth=2)
        
        wallfig, wallax = plt.subplots()
        
        wallax.add_collection(Wall_Collection)
        wallax.set_xlim(RR.min()-0.05,RR.max()+0.05)
        wallax.set_ylim(ZZ.min()-0.05,ZZ.max()+0.05)
        wallax.set_xlabel('Radial Coordinate (m)')
        wallax.set_ylabel('Vertical Coordinate (m)')
        wallax.set_aspect('equal')
        
        self.WS=Wall_Seg
        self.WC=Wall_Collection

    def plot2d_b2(self, vals, ax=None, scale='log', label='', lb=None, ub=None, **kwargs):
        '''Method to plot 2D fields on B2 grids. 
        Colorbars are set to be manually adjustable, allowing variable image saturation.

        Parameters
        ----------
        vals : array (self._getvar('ny'), self._getvar('nx'))
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

        if np.prod(vals.shape)==self._getvar('crx').shape[1]*self._getvar('crx').shape[2]:
            # Excluding boundary cells in plot2d_b2
            vals = vals[1:-1,1:-1]

        vals = vals.flatten()
        xx = self._getvar('crx').transpose(2,1,0)[1:-1,1:-1,:]
        yy = self._getvar('cry').transpose(2,1,0)[1:-1,1:-1,:]
        NY = self._getvar('ny'); NX = self._getvar('nx')

        # Avoid zeros that may derive from low Monte Carlo statistics
        if np.any(vals==0): vals[vals==0.0] = nsmallest(2,np.unique(vals))[1]
        
        patches = []
        for iy in np.arange(0,NY):
            for ix in np.arange(0,NX):
                rr = np.atleast_2d(xx[ix,iy,[0,1,3,2]]).T
                zz = np.atleast_2d(yy[ix,iy,[0,1,3,2]]).T
                patches.append(mpl.patches.Polygon(np.hstack((rr,zz)), True,linewidth=3))

        # collect al patches
        p = mpl.collections.PatchCollection(patches,False, edgecolor='k',linewidth=0.1, **kwargs)

        p.set_array(np.array(vals))

        if lb is None:
            lb = np.min(vals)
        if ub is None:
            ub = np.max(vals)

        if scale=='linear':
            p.set_clim([lb, ub])
        elif scale=='log':
            p.norm = mpl.colors.LogNorm(vmin=lb,vmax=ub)
        elif scale=='symlog':
            p.norm = mpl.colors.SymLogNorm(linthresh=ub/10.,base=10,
                                        linscale=0.5, vmin=lb,vmax=ub)
        else:
            raise ValueError('Unrecognized scale parameter')
        
        ax.add_collection(p)
        tickLocs = [ub,ub/10,lb/10,lb]
        cbar = plt.colorbar(p,ax=ax, pad=0.01, ticks = tickLocs if scale=='symlog' else None)
        
        cbar = plot_tools.DraggableColorbar(cbar,p)
        cbar.connect()
        
        #cbar.set_label(label)
        ax.set_title(label)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        #plt.grid(True)
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
        #cbar.set_label(label)
        ax.set_title(label)
        cbar = plot_tools.DraggableColorbar(cbar,cntr)
        cbar.connect()



    def get_radial_prof(self, vals, dz_mm=5, theta=0, label='', plot=False):
        '''Extract radial profiles of a quantity "quant" from the SOLPS run. 
        This function returns profiles on the low- (LFS) and high-field-side (HFS) midplane, 
        as well as flux surface averaged (FSA) ones. 

        Parameters
        ----------
        vals : array (self._getvar('ny'), self._getvar('nx'))
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
        
        if np.prod(vals.shape) == self._getvar('crx').shape[1]*self._getvar('crx').shape[2]:
            # Exclude boundary cells
            vals = vals[1:-1, 1:-1]

        rhop_2D = coords.get_rhop_RZ(self._getvar('R'), self._getvar('Z'), self.geqdsk)
        
        # evaluate FSA radial profile inside the LCFS
        def avg_function(r, z):
            if any(coords.get_rhop_RZ(r, z, self.geqdsk)<np.min(rhop_2D)):
                return np.nan
            else:
                return griddata((self._getvar('R').flatten(), self._getvar('Z').flatten()), vals.flatten(),
                                (r,z), method='linear')

        prof_FSA = self.geqdsk['fluxSurfaces'].surfAvg(function=avg_function)
        rhop_FSA = np.sqrt(self.geqdsk['fluxSurfaces']['geo']['psin'])

        # get R axes on the midplane on the LFS and HFS
        # rule-of-thumb to identify vertical resolution:
        _dz = (np.max(self._getvar('Z')) - np.min(self._getvar('Z')))/\
              ((self._getvar('nx')+self._getvar('ny'))/10.) 
        mask = (self._getvar('Z').flatten()>-_dz)&(self._getvar('Z').flatten()<_dz)
        R_midplane = self._getvar('R').flatten()[mask]
        
        R_midplane_lfs = R_midplane[R_midplane>self.geqdsk['RMAXIS']]
        _R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),1000)
        
        R_midplane_hfs = R_midplane[R_midplane<self.geqdsk['RMAXIS']]
        _R_HFS = np.linspace(np.min(R_midplane_hfs), np.max(R_midplane_hfs),1000)

        # get midplane radial profile...
        # ...on the LFS:
        _prof_LFS = griddata((self._getvar('R').flatten(),self._getvar('Z').flatten()), vals.flatten(),
                             (_R_LFS,0.5*dz_mm*1e-3*np.random.random(len(_R_LFS))),
                             #(_R_LFS,np.zeros_like(_R_LFS)),
                             method='linear')
        _prof_LFS[_prof_LFS<0]=np.nan
        R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),100)
        rhop_LFS = coords.get_rhop_RZ(R_LFS,np.zeros_like(R_LFS), self.geqdsk)

        # ... and on the HFS:
        _prof_HFS = griddata((self._getvar('R').flatten(),self._getvar('Z').flatten()), vals.flatten(), #self.quants[quant].flatten(),
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
        Z_core = self._getvar('Z')[self.unit_r:2*self.unit_r,self.unit_p:3*self.unit_p]
        R_core = self._getvar('R')[self.unit_r:2*self.unit_r,self.unit_p:3*self.unit_p]

        # find indices of poloidal grid nearest to Z=0 in the innermost radial shell
        midplane_LFS_idx = np.argmin(np.abs(Z_core[0,R_core[0,:]>self.geqdsk['RMAXIS']]))
        midplane_HFS_idx = np.argmin(np.abs(Z_core[0,R_core[0,:]<self.geqdsk['RMAXIS']]))

        # convert to indices on self._getvar('Z') and self._getvar('R')
        JXI = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]<self.geqdsk['RMAXIS']][midplane_HFS_idx]  # HFS_mid_pol_idx
        JXA = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]>self.geqdsk['RMAXIS']][midplane_LFS_idx] # LFS_mid_pol_idx

        # find rhop along midplane grid chords
        rhop_chord_HFS = coords.get_rhop_RZ(self._getvar('R')[:,JXI],self._getvar('Z')[:,JXI], self.geqdsk)
        rhop_chord_LFS = coords.get_rhop_RZ(self._getvar('R')[:,JXA],self._getvar('Z')[:,JXA], self.geqdsk)

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
        (rhop = 1 = LCFS by default) as a function of the poloidal angle theta

        Parameters
        ----------
        vals : array (self._getvar('ny'), self._getvar('nx'))
            Data array for a variable of interest.
        plot : bool
            If True, plot poloidal profile.
        label : string
            Label for plot
        rhop : float
            Radial coordinate, in rho_p, at which to take poloidal surface. Default is 1 (LCFS)
        topology : str
            Magnetic topology, one of ['USN','LSN']
            Note that double nulls ('DN') are not yet handled. 
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

        if np.prod(vals.shape)==self._getvar('crx').shape[1]*self._getvar('crx').shape[2]:
            # Exclude boundary cells
            vals = vals[1:-1,1:-1]         

        try:
            assert isinstance(topology, str) and topology!='DN'
        except AssertionError:
            raise AssertionError('Unrecognized topology!')

        # find x-point coordinates
        idx = 0 if topology=='LSN' else -1
        self.xpoint = sorted(zip(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS']),
                             key=lambda x: x[1])[idx]
        
        _x_point = self.xpoint

        _R_points=np.linspace(np.min(self._getvar('R')),np.max(self._getvar('R')), 202)
        
        if topology=='LSN':
            _Z_points=np.linspace(_x_point[1],np.max(self._getvar('Z')), 200)
        elif topology=='USN':
            _Z_points=np.linspace(np.min(self._getvar('Z')),_x_point[1], 200)
        elif topology=='DN':
            # not yet functional
            _Z_points=np.linspace(_x_point[0][1],_x_point[1][1],1200)
            _x_point=_x_point[0]
        else:
            raise ValueError('Unrecognized topology!')

        _R_grid,_Z_grid=np.meshgrid(_R_points,_Z_points,copy=False)

        rhop_2D = coords.get_rhop_RZ(_R_grid,_Z_grid, self.geqdsk)

        # rule-of-thumb to identify radial resolution:
        dr_rhop = (np.max(self._getvar('R')) - np.min(self._getvar('R')))/(self._getvar('ny')*10) 
        
        _mask=(rhop_2D<rhop+dr_rhop)&(rhop_2D>rhop-dr_rhop)
        
        _R_rhop=_R_grid[_mask]
        _Z_rhop=_Z_grid[_mask]

        # Need a way to get more resolution of the R and Z coordinates for given rhop
        prof_rhop = griddata((self._getvar('R').flatten(),self._getvar('Z').flatten()),
                             vals.flatten(), (_R_rhop,_Z_rhop), method='cubic')
        
        Rmaxis = self.geqdsk['RMAXIS']
        Zmaxis = self.geqdsk['ZMAXIS']
        
        _LFS_midplane_vect = np.array([np.max(self._getvar('R'))-Rmaxis, 0])
        _XP_vect = np.array([_x_point[0]-Rmaxis, _x_point[1]-Zmaxis])
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
        
        #num = 100 # arbitrary number, gives a fair level of resolution
        #num2 = np.floor_divide(len(poloidal_prof[0]), num)
        #
        #if num2>1:
        #    theta_prof = np.nanmedian(poloidal_prof[0][:num*num2].reshape(-1,num2),axis=1)
        #    pol_prof = np.nanmedian(poloidal_prof[1][:num*num2].reshape(-1,num2),axis=1)
        #else:
        #    # catch cases where there aren't enough points -- to be improved
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

    def _mdsmap(self):
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

        self.mdsmap = {
            # general
            'user': ident+'USER',
            'version': ident+'VERSION',
            'solpsversion': ident+'SOLPSVERSION',
            'directory': ident+'directory',
            'exp': ident+'EXP',
            
            # Dimensions
            'nx': snaptopdims+'NX',
            'ny': snaptopdims+'NY',
            'ns': snaptopdims+'NS',
            'natm': snaptopdims+'NATM',
            'nion': snaptopdims+'NION',
            'nmol': snaptopdims+'NMOL',
            
            # Indices
            'imp': snaptopdims+'IMP',
            'omp': snaptopdims+'OMP',
            'sep': snaptopdims+'SEP',
            'targ1': snaptopdims+'TARG1',
            'targ2': snaptopdims+'TARG2',
            'targ3': snaptopdims+'TARG3',
            'targ4': snaptopdims+'TARG4',
            'regflx': snaptopgrid+'REGFLX',
            'regfly': snaptopgrid+'REGFLY',
            'regvol': snaptopgrid+'REGVOL',

            # Neighbours
            'bottomix': snaptopgrid+'BOTTOMIX',
            'bottomiy': snaptopgrid+'BOTTOMIY',
            'leftix': snaptopgrid+'LEFTIX',
            'leftiy': snaptopgrid+'LEFTIY',
            'rightix': snaptopgrid+'RIGHTIX',
            'rightiy': snaptopgrid+'RIGHTIY',
            'topix': snaptopgrid+'TOPIX',
            'topiy': snaptopgrid+'TOPIY',
            
            # Grid and vessel geometry
            'R': snaptopgrid+'R',
            'Z': snaptopgrid+'Z',
            'cr': snaptopgrid+'CR',
            'crx': snaptopgrid+'CR_X', # R-coordinate of the cell x-face
            'cry': snaptopgrid+'CR_Y', # R-coordinate of the cell y-face
            'cz': snaptopgrid+'CZ',
            'czx': snaptopgrid+'CZ_X', # Z-coordinate of the cell x-face
            'czy': snaptopgrid+'CZ_Y',  # Z-coordinate of the cell y-face
            'hx': snaptop+'HX',
            'hy': snaptop+'HY',
            'hy1': snaptop+'HY1',
            'dspar': snaptopgrid+'DSPAR',
            'dspol': snaptopgrid+'DSPOL',
            'dsrad': snaptopgrid+'DSRAD',
            'sx': snaptop+'SX',
            'sxperp': snaptop+'SXPERP',
            'sy': snaptop+'SY',
            'vessel': snaptopgrid+'VESSEL',
            
            # Plasma characterization
            'am': snaptopgrid+'AM',
            'za': snaptopgrid+'ZA',
            'zn': snaptopgrid+'ZN',
            'bb': snaptop+'B',
            'po': snaptop+'PO',
            'pot': snaptopgrid+'POT',
            'poti': snaptopgrid+'POTI',
            'qz': snaptop+'QZ',
            'visc': snaptop+'VISC',
            'vol': snaptop+'VOL',
            
            # Densities
            'ne': snaptop+'NE',
            'na': snaptop+'NA',
            'dab2': snaptop+'DAB2',
            'dmb2': snaptop+'DMB2',
            'dib2': snaptop+'DIB2',
            
            # Temperatures
            'te': snaptop+'TE',
            'ti': snaptop+'TI',
            'tab2': snaptop+'TAB2',
            'tmb2': snaptop+'TMB2',
            'tib2': snaptop+'TIB2',
            
            # Velocities
            'ua': snaptop+'UA',
            'vlax': snaptop+'VLAX',
            'vlay': snaptop+'VLAY',
            
            # Fluxes
            'fchx': snaptop+'FCHX',
            'fchy': snaptop+'FCHY',
            'fhex': snaptop+'FHEX',
            'fhey': snaptop+'FHEY',
            'fhix': snaptop+'FHIX',
            'fhiy': snaptop+'FHIY',
            'fhjx': snaptop+'FHJX', # (not listed in the manual)
            'fhjy': snaptop+'FHJY', # (not listed in the manual)
            'fhmx': snaptop+'FHMX',
            'fhmy': snaptop+'FHMY',
            'fhpx': snaptop+'FHPX',
            'fhpy': snaptop+'FHPY',
            'fhtx': snaptop+'FHTX',
            'fhty': snaptop+'FHTY',
            'fmox': snaptop+'FMOX',
            'fmoy': snaptop+'FMOY',
            'fnax': snaptop+'FNAX',
            'fnax_32': snaptop+'FNAX_32',
            'fnax_52': snaptop+'FNAX_52',
            'fnay': snaptop+'FNAY',
            'fnay_32': snaptop+'FNAY_32',
            'fnay_52': snaptop+'FNAY_52',
            'pefa': snaptop+'PEFA',
            'pefm': snaptop+'PEFM',
            'pfla': snaptop+'PFLA',
            'pflm': snaptop+'PFLM',
            'refa': snaptop+'REFA',
            'refm': snaptop+'REFM',
            'rfla': snaptop+'RFLA',
            'rflm': snaptop+'RFLM',
            
            # Coefficients
            'alf': snaptop+'ALF',
            'dna0': snaptop+'D',
            'dp': snaptop+'DP',
            'kye': snaptop+'KYE',
            'kyi': snaptop+'KYI',
            'kyi0': snaptop+'KYI0',
            'sig': snaptop+'SIG',
            'rpt': snaptop+'RPT', # (not listed in the manual)
            
            # Rates, losses and sources
            'rcxhi': snaptop+'RCXHI',
            'rcxmo': snaptop+'RCXMO',
            'rcxna': snaptop+'RCXNA',
            'rqahe': snaptop+'RQAHE',
            'rqbrm': snaptop+'RQBRM',
            'rqrad': snaptop+'RQRAD',
            'rrahi': snaptop+'RRAHI',
            'rramo': snaptop+'RRAMO',
            'rrana': snaptop+'RRANA',
            'rsahi': snaptop+'RSAHI',
            'rsamo': snaptop+'RSAMO',
            'rsana': snaptop+'RSANA',
            
            'smo': snaptop+'SMO',
            'smq': snaptop+'SMQ',
            'b2npmo_smav': snaptop+'SMAV',
            'resmo': snaptop+'RESMO',
            
            # Texts
            'textan': snaptop+'TEXTAN',
            'textcomp': snaptop+'TEXTCOMP',
            'textin': snaptop+'TEXTIN',
            'textmn': snaptop+'TEXTMN',
            'textpl': snaptop+'TEXTPL',

            # Targets
            'TARGET1_ds': timedept1+'DS',
            'TARGET1_ft': timedept1+'FT',
            'TARGET1_fe': timedept1+'FE',
            'TARGET1_fi': timedept1+'FI',
            'TARGET1_fc': timedept1+'FC',
            'TARGET1_te': timedept1+'TE',
            'TARGET1_ti': timedept1+'TI',
            'TARGET1_ne': timedept1+'NE',
            'TARGET1_po': timedept1+'PO',
            
            'TARGET2_ds': timedept2+'DS',
            'TARGET2_ft': timedept2+'FT',
            'TARGET2_fe': timedept2+'FE',
            'TARGET2_fi': timedept2+'FI',
            'TARGET2_fc': timedept2+'FC',
            'TARGET2_te': timedept2+'TE',
            'TARGET2_ti': timedept2+'TI',
            'TARGET2_ne': timedept2+'NE',
            'TARGET2_po': timedept2+'PO',
            
            'TARGET3_ds': timedept3+'DS',
            'TARGET3_ft': timedept3+'FT',
            'TARGET3_fe': timedept3+'FE',
            'TARGET3_fi': timedept3+'FI',
            'TARGET3_fc': timedept3+'FC',
            'TARGET3_te': timedept3+'TE',
            'TARGET3_ti': timedept3+'TI',
            'TARGET3_ne': timedept3+'NE',
            'TARGET3_po': timedept3+'PO',
            
            'TARGET4_ds': timedept4+'DS',
            'TARGET4_ft': timedept4+'FT',
            'TARGET4_fe': timedept4+'FE',
            'TARGET4_fi': timedept4+'FI',
            'TARGET4_fc': timedept4+'FC',
            'TARGET4_te': timedept4+'TE',
            'TARGET4_ti': timedept4+'TI',
            'TARGET4_ne': timedept4+'NE',
            'TARGET4_po': timedept4+'PO',
            
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

    def find_gfile(self):
        '''Identify the name of the gEQDSK file within the SOLPS output directory.
        '''
        for filename in os.listdir(self.path+os.sep+'baserun'):
            if filename.startswith('g'): # assume only 1 file starts with 'g'
                return filename



        
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


