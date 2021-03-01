'''Aurora tools to read SOLPS results and extract atomic neutral density from EIRENE output.
These enable examination of charge exchange recombination and radiation due to the interaction 
of heavy ions and thermal neutrals. 

F.Sciortino & R.Reksoatmodjo, 2021
'''
import matplotlib.pyplot as plt
import os
import numpy as np
plt.ion()
from scipy.interpolate import griddata, interp1d
import matplotlib as mpl
import matplotlib.tri as tri
from heapq import nsmallest
import warnings
from scipy import constants
from . import plot_tools
from . import coords


class solps_case:
    def __init__(self, path, geqdsk, solps_run='',
                 case_num=0, form='extracted', extracted_labels=None):
        '''Read SOLPS output and prepare for Aurora impurity-neutral analysis. 

        Parameters
        ----------
        path : str
            Path to output files. If these are "extracted" files from SOLPS (form='extracted'),
            then this is the path to the disk location where each of the required files
            can be found. 
            If form='full', this path indicates where to find the directory named "baserun"
            and the 'solps_run' one.
        geqdsk : str or `omfit_classes.omfit_geqdsk.OMFITgeqdsk` class instance
            Path to the geqdsk to load from disk, or instance of the `omfit_classes.omfit_geqdsk.omfit_geqdsk` 
            class that contains the processed gEQDSK file already. 
        solps_run : str
            If form='full', this string specifies the directory (relative to the given path)
            where case-specific files for a SOLPS run can be found (e.g. 'b2fstate').
        case_num : int
            Index/integer identifying the SOLPS case of interest. 
        form : str
            Form of SOLPS output to be loaded, one of {'full','extracted'}. 
            The 'full' output consists of 'b2fstate', 'b2fgmtry', 'fort.44', etc.
            The 'extracted' output consists of individual files containing one quantity each.
            If form='extracted', the 'extracted_labels' argument allows to specify files 
            nomanclature. 
        extracted_labels : dict
            Only used if form='extracted', this dictionary allows specification of the names of
            files/variables to be extracted. Default is to use
            [RadLoc, VertLoc, Ne, Te, NeuDen, NeuTemp, MolDen, MolTemp, Ti]
            each of which will be expected to have "case_num" appended to the name of the file.
        '''

        self.path = path
        self.solps_run = solps_run
        self.case_num = case_num
        self.form = form
        
        if isinstance(geqdsk, str): # user passed a path to a gfile on disk
            from omfit_classes import omfit_eqdsk # import omfit_eqdsk here to avoid issues with docs and packaging
            self.geqdsk = omfit_eqdsk.OMFITgeqdsk(geqdsk)
        else:
            self.geqdsk = geqdsk

        self.labels = {'ne':r'$n_e$ [$m^{-3}$]' , 'Te':r'$T_e$ [eV]',
                      'nn':r'$n_n$ [$m^{-3}$]' , 'Tn':r'$T_n$ [eV]',
                      'nm':r'$n_m$ [$m^{-3}$]' , 'Tm':r'$T_m$ [eV]',
                      'Ti':r'$T_i$ [eV]',
            }
            
        if form=='extracted':
            self.ext_names_map = extracted_labels
            if self.ext_names_map is None: # set default map to identify extracted data file names
                self.ext_names_map = {'ne':'Ne','Te':'Te','nn':'NeuDen','Tn':'NeuTemp',
                                      'nm':'MolDen','Tm':'MolTemp','Ti':'Ti',
                                      'R':'RadLoc','Z':'VertLoc'}

            tmp = np.loadtxt(path+f'/{self.ext_names_map["R"]}{case_num}')

            self.ny = int(np.max(tmp[:,1]))
            self.nx = int(np.max(tmp[:,0]))

            self.unit_p = self.nx//4
            self.unit_r = self.ny//2

            # Currently only set up for single null shapes
            self.double_null = False

            # Obtain indices for chosen radial regions
            _R_idxs = np.array([],dtype=int)
            _R_idxs = np.concatenate((_R_idxs, np.arange(self.unit_r+1)))  # PFR and core
            self.R_idxs = np.concatenate((_R_idxs, np.arange(self.unit_r+1,2*self.unit_r+2))) # open-SOL
            
            # obtain indices for chosen poloidal regions
            _P_idxs = np.array([],dtype=int)
            _P_idxs = np.concatenate((_P_idxs, np.arange(self.unit_p+1)))  # Inner PFR
            _P_idxs = np.concatenate((_P_idxs, np.arange(self.unit_p+1,3*self.unit_p+1)))  # core/open SOL
            self.P_idxs = np.concatenate((_P_idxs, np.arange(3*self.unit_p+1,4*self.unit_p+2)))  # outer PFR

            
        elif form=='full':
            from omfit_classes import omfit_solps # import omfit_solps here to avoid issues with docs and packaging
            self.b2fstate = omfit_solps.OMFITsolps(self.path+os.sep+self.solps_run+os.sep+'b2fstate')
            self.geom = omfit_solps.OMFITsolps(path+os.sep+'baserun'+os.sep+'b2fgmtry')

            self.nx,self.ny = self.geom['nx,ny']
            
            # (0,:,:): lower left corner, (1,:,:): lower right corner
            # (2,:,:): upper left corner, (3,:,:): upper right corner.
            
            self.crx = self.geom['crx'].reshape(4,self.ny+2,self.nx+2)  # horizontal 
            self.cry = self.geom['cry'].reshape(4,self.ny+2,self.nx+2)  # vertical

            # figure out if single or double null
            self.double_null = self.cry.shape[2]%4==0
            
            # uncertain units for splitting of radial and poloidal regions...
            self.unit_p = (self.crx.shape[2]-4 if self.double_null else self.crx.shape[2])//4
            self.unit_r = (self.crx.shape[1]-2)//2   #(self.crx.shape[1]-2 if self.double_null else self.crx.shape[1])//2 

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

        # load data arrays
        self.load_data() #P_idxs=self.P_idxs, R_idxs=self.R_idxs)  # TODO: enable subselection of regions
        
        # set zero densities equal to min to avoid log issues
        #self.quants['nn'][self.quants['nn']==0.0] = nsmallest(2,np.unique(self.quants['nn'].flatten()))[1]

        
    def load_data(self, fields=None, P_idxs=None, R_idxs=None,
                  Rmin=None, Rmax=None, Pmin=None, Pmax=None):
        '''Load SOLPS output for each of the needed quantities ('extracted' form)

        Parameters
        -----------------
        fields : list or array
            List of fields to fetch from SOLPS output. If left to None, by default uses
            self.labels.keys(), i.e. ['ne','Te','nn','Tn','nm','Tm','Ti']
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
            fields = list(self.labels.keys())
            
        self.quants = quants = {}
        
        if self.form=='extracted':
            # "extracted" format has each variable in a different file
            
            self.R = np.atleast_2d(
                np.loadtxt(self.path+f'/{self.ext_names_map["R"]}{self.case_num}',
                           usecols=(3)).reshape((self.ny+2,self.nx+2))[:,P_idxs])[R_idxs,:]
            self.Z = np.atleast_2d(
                np.loadtxt(self.path+f'/{self.ext_names_map["Z"]}{self.case_num}',
                           usecols=(3)).reshape((self.ny+2,self.nx+2))[:,P_idxs])[R_idxs,:]

            for field in fields:
                tmp =  np.loadtxt(self.path+f'/{self.ext_names_map[field]}{self.case_num}',
                                  usecols=(3)).reshape((self.ny+2,self.nx+2))
                quants[field] = np.atleast_2d(tmp[:,P_idxs])[R_idxs,:]

    
        elif self.form=='full':

            # eliminate end (buffer) points of grid
            self.R = np.mean(self.crx,axis=0)[1:-1,1:-1][R_idxs,:][:,P_idxs]
            self.Z = np.mean(self.cry,axis=0)[1:-1,1:-1][R_idxs,:][:,P_idxs]

            self.fort44 = self.load_fort44()
            self.fort46 = self.load_fort46()
            
            quants['ne'] = self.b2fstate['ne'][1:-1,1:-1][R_idxs,:][:,P_idxs] # m^-3
            #quants['nn'] = self.b2fstate['na'][0,R_idxs,:][:,P_idxs] # m^-3   # only neutral component
            nn44 = self.fort44['dab2'][:,:,0].T
            nn44[nn44==0.0] = nsmallest(2,np.unique(nn44.flatten()))[1]
            quants['nn'] = nn44[R_idxs,:][:,P_idxs] # m^-3   # only neutral component
            quants['Tn'] = self.fort44['tab2'][:,:,0].T[R_idxs,:][:,P_idxs]/constants.e # eV
            quants['Te'] = self.b2fstate['te'][1:-1,1:-1][R_idxs,:][:,P_idxs]/constants.e # eV
            quants['Ti'] = self.b2fstate['ti'][1:-1,1:-1][R_idxs,:][:,P_idxs]/constants.e # eV
            quants['nm'] = self.fort44['dmb2'][:,:,0].T[R_idxs,:][:,P_idxs] # D molecular density
            quants['Tm'] = self.fort44['tmb2'][:,:,0].T[R_idxs,:][:,P_idxs] # D molecular temperature
            
            # EIRENE nodes and triangulation
            self.xnodes, self.ynodes, self.triangles = self.load_eirene_mesh()


            
            
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
        with open(self.path +os.sep+self.solps_run+os.sep+'fort.44', 'r') as f:
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
            out[key] = np.array(out[key]).reshape(*fort44_info[key][1],order='F')
            
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
        with open(self.path +os.sep+self.solps_run+os.sep+'fort.46', 'r') as f:
            contents = f.readlines()

        out['DIM']=contents[0].split()[0]

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
            if key in ['species','DIM']:
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

                
    def plot2d_b2(self, vals, ax=None, scale='log', label='', lb=None, ub=None, **kwargs):
        '''Method to plot 2D fields on B2 grids. 
        Colorbars are set to be manually adjustable, allowing variable image saturation.

        Parameters
        ----------
        vals : array (self.ny, self.nx)
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

        if np.prod(vals.shape)==self.crx.shape[1]*self.crx.shape[2]:
            #print('Excluding boundary cells in plot2d_b2')
            vals = vals[1:-1,1:-1]

        vals = vals.flatten()
        xx = self.crx.transpose(2,1,0)[1:-1,1:-1,:]
        yy = self.cry.transpose(2,1,0)[1:-1,1:-1,:]
        NY = self.ny; NX = self.nx

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
        plt.grid(True)
        ax.axis('scaled')

        


    def plot2d_eirene(self, vals,  ax = None, scale='log', label='', lb=None, ub=None,**kwargs):
        '''Method to plot 2D fields from EIRENE.

        Parameters
        ----------
        vals : array (self.ny, self.nx)
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
        kwargs
            Additional keyword arguments passed to the `tripcolor` function.

        '''
        # Avoid zeros that may derive from low Monte Carlo statistics
        if np.any(vals==0): vals[vals==0.0] = nsmallest(2,np.unique(vals.flatten()))[1]

        if ax is None:
            fig,ax = plt.subplots(figsize=(8,11))

        # plot LCFS
        ax.plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
        
        if lb is None:
            lb = np.min(vals)
        if ub is None:
            ub = np.max(vals)
        
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
        cbar = plt.colorbar(cntr, format='%.3g', ax=ax)
        #cbar.set_label(label)
        ax.set_title(label)
        cbar = plot_tools.DraggableColorbar(cbar,cntr)
        cbar.connect()



    def get_radial_prof(self, quant='nn', dz_mm=5, plot=False):
        '''Extract radial profiles of a quantity "quant" from the SOLPS run. 
        This function returns profiles on the low- (LFS) and high-field-side (HFS) midplane, 
        as well as flux surface averaged (FSA) ones. 

        Parameters
        ----------
        quant : str
            Quantity of interest. Default is 'nn' (neutral atomic density). See self.labels.keys()
            for other options.
        dz_mm : float
            Vertical range [mm] over which quantity should be averaged near the midplane. 
            Mean and standard deviation of profiles on the LFS and HFS will be returned based on
            variations of atomic neutral density within this vertical span.
            Note that this does not apply to the FSA calculation. Default is 5 mm.
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

        rhop_2D = coords.get_rhop_RZ(self.R,self.Z, self.geqdsk)
        
        # evaluate FSA radial profile inside the LCFS
        def avg_function(r,z):
            if any(coords.get_rhop_RZ(r,z, self.geqdsk)<np.min(rhop_2D)):
                return np.nan
            else:
                return griddata((self.R.flatten(),self.Z.flatten()), self.quants[quant].flatten(),
                                (r,z), method='linear')

        prof_FSA = self.geqdsk['fluxSurfaces'].surfAvg(function=avg_function)
        rhop_FSA = np.sqrt(self.geqdsk['fluxSurfaces']['geo']['psin'])
        
        # get R axes on the midplane on the LFS and HFS
        _dz = (np.max(self.Z) - np.min(self.Z))/((self.nx+self.ny)/10.) # rule-of-thumb to identify vertical resolution
        mask = (self.Z.flatten()>-_dz)&(self.Z.flatten()<_dz)
        R_midplane = self.R.flatten()[mask]
        
        R_midplane_lfs = R_midplane[R_midplane>self.geqdsk['RMAXIS']]
        _R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),1000)
        
        R_midplane_hfs = R_midplane[R_midplane<self.geqdsk['RMAXIS']]
        _R_HFS = np.linspace(np.min(R_midplane_hfs), np.max(R_midplane_hfs),1000)

        # get midplane radial profile...
        # ...on the LFS:
        _prof_LFS = griddata((self.R.flatten(),self.Z.flatten()), self.quants[quant].flatten(),
                             (_R_LFS,0.5*dz_mm*1e-3*np.random.random(len(_R_LFS))),
                             #(_R_LFS,np.zeros_like(_R_LFS)),
                             method='linear')
        _prof_LFS[_prof_LFS<0]=np.nan
        R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),100)
        rhop_LFS = coords.get_rhop_RZ(R_LFS,np.zeros_like(R_LFS), self.geqdsk)

        # ... and on the HFS:
        _prof_HFS = griddata((self.R.flatten(),self.Z.flatten()), self.quants[quant].flatten(),
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

        # now obtain also the simple poloidal grid slice near the midplane (LFS and HFS)
        # These are commonly used for SOLPS analysis, using the JXA and JXI indices (which we re-compute here)
        Z_core = self.Z[self.unit_r:2*self.unit_r,self.unit_p:3*self.unit_p]
        R_core = self.R[self.unit_r:2*self.unit_r,self.unit_p:3*self.unit_p]

        # find indeces of poloidal grid nearest to Z=0 in the innermost radial shell
        midplane_LFS_idx = np.argmin(np.abs(Z_core[0,R_core[0,:]>self.geqdsk['RMAXIS']]))
        midplane_HFS_idx = np.argmin(np.abs(Z_core[0,R_core[0,:]<self.geqdsk['RMAXIS']]))

        # convert to indices on self.Z and self.R
        JXI = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]<self.geqdsk['RMAXIS']][midplane_HFS_idx]  # HFS_mid_pol_idx
        JXA = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]>self.geqdsk['RMAXIS']][midplane_LFS_idx] # LFS_mid_pol_idx

        # find rhop along midplane grid chords
        rhop_chord_HFS = coords.get_rhop_RZ(self.R[:,JXI],self.Z[:,JXI], self.geqdsk)
        rhop_chord_LFS = coords.get_rhop_RZ(self.R[:,JXA],self.Z[:,JXA], self.geqdsk)

        if plot:
            # plot quantties on linear scale -- need to adapt labels
            #lab = fr'${self.labels[quant].split("$")[1]}$ [${self.labels[quant].split("$")[3]}$]'
            
            # compare FSA radial profiles with midplane (LFS and HFS) ones
            fig,ax = plt.subplots()
            ax.plot(rhop_FSA, prof_FSA, label='FSA')
            ax.plot(rhop_LFS, prof_LFS, label='LFS midplane')
            ax.plot(rhop_HFS, prof_HFS, label='HFS midplane')
            ax.plot(rhop_chord_LFS, self.quants[quant][:,JXA], label='LFS grid midplane')
            ax.plot(rhop_chord_HFS, self.quants[quant][:,JXI], label='HFS grid midplane')
            ax.set_xlabel(r'$\rho_p$')
            ax.set_ylabel(self.labels[quant]) #lab)
            ax.legend(loc='best').set_draggable(True)
            plt.tight_layout()

        return rhop_FSA, prof_FSA, rhop_LFS, prof_LFS, rhop_HFS, prof_HFS


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
        'wldnek': [r'Heat transferred by neutrals (W), total over strata', (NLIM+NSTS)],
        'wldnep': [r'Potential energy released by neutrals (W), total over strata', (NLIM+NSTS)],
        'wldna': [r'Flux of atoms impinging on surface (A), total over strata', (NLIM+NSTS,NATM)],
        'ewlda': [r'Average energy of impinging atoms on surface (eV), total over strata', (NLIM+NSTS,NATM)],
        'wldnm': [r'Flux of molecules impinging on surface (A), total over strata', (NLIM+NSTS,NMOL)],
        'ewldm': [r'Average energy of impinging molecules on surface (eV), total over strata', (NLIM+NSTS,NMOL)],
        'p1,p2': [r'Endpoints of surface (X and Y coordinates, in m), total over strata', (NLIM)],
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
            'wlpump_res(M)': [r' Pumped flux per molecule (A)', (NCL, NMOL)],
            'wlpump_res(I)': [r' Pumped flux per test ion (A)', (NCL, NION)],
            'wlpump_res(P)': [r' Pumped flux per plasma ion (A)', (NCL, NPLS)],
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

    # extra, undocumente?
    #fort44_info.update({'wall_geometry' : [r'Wall geometry points', ((NSTRA+1)*(5*NSTS+1),)]})  # check dimensions
    fort44_info.update({'wall_geometry' : [r'Wall geometry points', (4*NLIM,)]})  # check dimensions
    return fort44_info
