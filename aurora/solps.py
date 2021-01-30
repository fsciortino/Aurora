'''Aurora tools to read SOLPS results and extract atomic neutral density from EIRENE output.
These enable examination of charge exchange recombination and radiation due to the interaction 
of heavy ions and thermal neutrals. 

F.Sciortino & R.Reksoatmodjo, 2021
'''
import matplotlib.pyplot as plt
import os
import numpy as np
plt.ion()
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.tri as tri
from heapq import nsmallest
import warnings

from . import plot_tools
from . import coords

class solps_case:
    def __init__(self, path, geqdsk, solps_run='',
                 case_num=0, form='extracted', extracted_labels=None):
        '''Read SOLPS output and prepare for Aurora impurity-neutral analysis. 

        Args:
            path : str
                Path to output files. If these are "extracted" files from SOLPS (form='extracted'),
                then this is the path to the disk location where each of the required files
                can be found. These are currently expected to be named
                [RadLoc, VertLoc, Ne, Te, NeuDen, NeuTemp, MolDen, MolTemp, Ti]{case_num}
                Otherwise, if form='full', this path indicates where to find the directory named "baserun"
                and the 'solps_run' one.
            geqdsk : str or `omfit_geqdsk` class instance
                Path to the geqdsk to load from disk, or instance of the `omfit_geqdsk` class that
                contains the processed gEQDSK file already. 

        Keyword Args:
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
            import omfit_eqdsk # import omfit_eqdsk here to avoid issues with docs and packaging
            self.geqdsk = omfit_eqdsk.OMFITgeqdsk(geqdsk)
        else:
            self.geqdsk = geqdsk

        self.labels = {'ne':r'log($n_e$) [$m^{-3}$]' , 'Te':r'log($T_e$) [eV]',
                      'nn':'log($n_n$) [$m^{-3}$]' , 'Tn':'log($T_n$) [eV]',
                      'nm':'log($n_m$) [$m^{-3}$]' , 'Tm':'log($T_m$) [eV]',
                      'Ti':r'log($T_i$) [eV]',
            }
            
        if form=='extracted':
            self.ext_names_map = extracted_labels
            if self.ext_names_map is None: # set default map to identify extracted data file names
                self.ext_names_map = {'ne':'Ne','Te':'Te','nn':'NeuDen','Tn':'NeuTemp',
                                      'nm':'MolDen','Tm':'MolTemp','Ti':'Ti',
                                      'R':'RadLoc','Z':'VertLoc'}

            tmp = np.loadtxt(path+f'/{self.ext_names_map["R"]}{case_num}')
            self.unit_p = int(np.max(tmp[:,0])//4)
            self.unit_r = int(np.max(tmp[:,1])//2)

            self.nx = int(np.max(tmp[:,1]))
            self.ny = int(np.max(tmp[:,0]))
            
        elif form=='full':
            import omfit_solps # import omfit_solps here to avoid issues with docs and packaging
            self.b2fstate = omfit_solps.OMFITsolps(path+os.sep+solps_run+os.sep+'b2fstate')
            self.geom = omfit_solps.OMFITsolps(path+os.sep+'baserun'+os.sep+'b2fgmtry')

            self.nx,self.ny = self.geom['nx,ny']
            
            # (0,:,:): lower left corner, (1,:,:): lower right corner
            # (2,:,:): upper left corner, (3,:,:): upper right corner.
            
            self.crx = self.geom['crx'].reshape(4,self.ny+2,self.nx+2)  # horizontal 
            self.cry = self.geom['cry'].reshape(4,self.ny+2,self.nx+2)  # vertical
            
            # uncertain units for splitting of radial and poloidal regions...
            self.unit_p = (self.cry.shape[2])//4
            self.unit_r = (self.crx.shape[1])//2 - 1
    
        # Obtain indices for chosen radial regions
        R_idxs = np.array([],dtype=int)
        R_idxs = np.concatenate((R_idxs, np.arange(self.unit_r+1)))  # PFR and core
        self.R_idxs = np.concatenate((R_idxs, np.arange(self.unit_r+1,2*self.unit_r+2))) # open-SOL

        # obtain indices for chosen poloidal regions
        P_idxs = np.array([],dtype=int)
        P_idxs = np.concatenate((P_idxs, np.arange(self.unit_p)))  # Inner PFR
        P_idxs = np.concatenate((P_idxs, np.arange(self.unit_p,3*self.unit_p)))  # core/open SOL
        self.P_idxs = np.concatenate((P_idxs, np.arange(3*self.unit_p,4*self.unit_p)))  # outer PFR


    def load_data(self, fields=None, P_idxs=None, R_idxs=None,
                  Rmin=None, Rmax=None, Pmin=None, Pmax=None):
        '''Load SOLPS output for each of the needed quantities ('extracted' form)

        Keyword Args:
            fields : list or array
                List of fields to fetch from SOLPS output. If left to None, by default uses
                self.labels.keys(), i.e. ['ne','Te','nn','nT','nm','Tm','Ti']
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
        
        Returns:
            quants : dict
                Dictionary containing 'R','Z' coordinates for 2D maps of each field requested by user.
        '''
        if P_idxs is None:
            if Pmax is None: Pmax = self.ny
            if Pmin is None: Pmin = 0
            P_idxs = np.arange(Pmin,Pmax)
        else:
            pass # user provided list of poloidal grid indices

        if R_idxs is None:
            if Rmax is None: Rmax = self.nx
            if Rmin is None: Rmin = 0
            R_idxs = np.arange(Rmin,Rmax)
        else:
            pass # user provided list of radial grid indices

        if fields is None:
            fields = list(self.labels.keys())
            
        self.quants = quants = {}
        
        if self.form=='extracted':
            self.R = np.atleast_2d(
                np.loadtxt(self.path+f'/{self.ext_names_map["R"]}{self.case_num}',
                           usecols=(3)).reshape((self.nx+2,self.ny+2))[R_idxs,:])[:,P_idxs+1]
            self.Z = np.atleast_2d(
                np.loadtxt(self.path+f'/{self.ext_names_map["Z"]}{self.case_num}',
                           usecols=(3)).reshape((self.nx+2,self.ny+2))[R_idxs,:])[:,P_idxs+1]

            for field in fields:
                tmp =  np.loadtxt(self.path+f'/{self.ext_names_map[field]}{self.case_num}',
                                  usecols=(3)).reshape((self.nx+2,self.ny+2))
                quants[field] = np.atleast_2d(tmp[R_idxs,:])[:,P_idxs+1]

            self.triang_b2 = tri.Triangulation(self.R.flatten(), self.Z.flatten())
            self.triang_eirene = self.triang_b2 # triangulation for EIRENE is mapped to b2 grid
    
        elif self.form=='full':

            # eliminate end (buffer) points of grid
            self.R = np.mean(self.crx,axis=0)[1:-1,:][:,1:-1] #[R_idxs,:][:,P_idxs]
            self.Z = np.mean(self.cry,axis=0)[1:-1,:][:,1:-1] #[R_idxs,:][:,P_idxs]

            for field in field_labels:
                if field in ['ne','Te','Ti']:
                    quants[field] = self.b2fstate[field][1:-1,:][:,1:-1] #[R_idxs,:][:,P_idxs]

            self.eirene_out = self.load_eirene_output(files=['fort.44'])  #'fort.46'

            ########
            ## Read on EIRENE mesh
            # molecular density and temperature
            # quants['nm'] = self.eirene_out['fort.46']['pdenm']
            # quants['Tm']= self.eirene_out['fort.46']['edenm']

            # # group atomic neutral density from all H isotopes
            # spmask = self.b2fstate['zn']==1
            # quants['nn'] = np.sum(
            #     self.eirene_out['fort.46']['pdena'].reshape(-1,sum(spmask)),
            #     axis=1)

            # quants['Tn'] = np.sum(
            #     self.eirene_out['fort.46']['edena'].reshape(-1,sum(spmask)),
            #     axis=1)
            ####

            tmp = self.eirene_out['fort.44']['dmb2'].reshape((self.nx,self.ny))
            quants['nm'] = np.atleast_2d(tmp).T #[R_idxs[:-2],:][:,P_idxs[:-2]]
            tmp = self.eirene_out['fort.44']['tmb2'].reshape((self.nx,self.ny))
            quants['Tm'] = np.atleast_2d(tmp).T #[R_idxs[:-2],:][:,P_idxs[:-2]]

            # group atomic neutral density from all H isotopes
            spmask = self.b2fstate['zn']==1
            tmp = np.sum(
                self.eirene_out['fort.44']['dab2'].reshape(-1,sum(spmask)),
                axis=1).reshape((self.nx,self.ny))
            quants['nn'] = np.atleast_2d(tmp).T #[R_idxs[:-2],:][:,P_idxs[:-2]]
            tmp = np.sum(
                self.eirene_out['fort.44']['tab2'].reshape(-1,sum(spmask)),
                axis=1).reshape((self.nx,self.ny))
            quants['Tn'] = np.atleast_2d(tmp).T #[R_idxs[:-2],:][:,P_idxs[:-2]]
            
            self.triang_b2 = tri.Triangulation(self.R.flatten(), self.Z.flatten())
            self.triang_eirene = self.eirene_out['triang']
            
            
            


    def load_eirene_output(self, files = ['fort.44','fort.46']):
        '''Load result from one of the fort.* files with EIRENE output, 
        as indicated by the "files" list argument.

        Keyword Args:
            files : list or array-like
                EIRENE output files to read. Default is to load all
                files for which this method has been tested. 

        Returns:
            eirene_out : dict
                Dictionary for each loaded file containing a subdictionary
                with keys for each loaded field from each file. 

        '''
        eirene_out = {}
        
        for filename in files: 
            eirene_out[filename] = {}
            # load each of these files into dictionary structures
            with open(self.path +os.sep+self.solps_run+os.sep+filename, 'r') as f:
                contents = f.readlines()
            ii=6

            while ii<len(contents[ii:]):
                if  contents[ii].startswith('*eirene'):
                    key = contents[ii].split()[3]
                    eirene_out[filename][key] = []
                else:
                    [eirene_out[filename][key].append(float(val)) for val in contents[ii].strip().split()]
                ii+=1

            for key in eirene_out[filename]:
                eirene_out[filename][key] = np.array(eirene_out[filename][key])

        # Now load fort.33 file with EIRENE nodes and cells
        Nodes=np.fromfile(self.path+os.sep+'baserun'+os.sep+'fort.33',sep=' ')
        NN=int(Nodes[0])
        XNodes=Nodes[1:NN+1]/100  # cm -->m
        YNodes=Nodes[NN+1:]/100
        
        # EIRENE triangulation
        Triangles = np.loadtxt(
            self.path+os.sep+'baserun'+os.sep+'fort.34',skiprows=1, usecols=(1,2,3))

        eirene_out['triang'] =tri.Triangulation(XNodes,YNodes,triangles=(Triangles-1))

        return eirene_out


    def process_solps_data(self, fields=None, P_idxs=None, R_idxs=None, plot=False):
        '''Load and process SOLPS to permit clear plotting. 
        
        Keyword Args:
            fields : dict
                Dictionary containing SOLPS outputs to process. Keys indicate the quantity, value its label
                (only used for plotting). If left to None, defaults fields of 'ne','Te','nn' and 'Tn' are used.
            P_idxs : list or array
                Poloidal indices to load.
            R_idxs : list or array
                Radial indices to load.
            plot : bool
                If True, plot results for all loaded 2D quantities. 

        Returns:
            quants : dict
                Dictionary containing 'R','Z' coordinates for 2D maps of each field requested by user.
                Quantities are processed and masked to facilitate plotting.
        '''
        if fields is None:
            fields = ['ne','Te','nn','Tn'] #list(self.labels.keys())

        if P_idxs is None:
            P_idxs = self.P_idxs
        if R_idxs is None:
            R_idxs = self.R_idxs

        # load data arrays
        self.load_data(fields = fields, P_idxs=P_idxs, R_idxs=R_idxs)

        # apply masking based on maximum side-length
        self.max_mask_len = 10* (np.max(self.Z)-np.min(self.Z))/(self.nx) #ROT to get an appropriate edge masking length
        self.triang_b2 = apply_mask(self.triang_b2, self.geqdsk, max_mask_len=self.max_mask_len)

        # set zero densities equal to min to avoid log issues
        self.quants['nn'][self.quants['nn']==0.0] = nsmallest(2,np.unique(self.quants['nn'].flatten()))[1]

        if plot:

            fig,axs = plt.subplots(2,2, figsize=(9,12),sharex=True) 
            ax = axs.flatten()

            cbars = [];
            for ii,field in enumerate(fields):
                label = fields[field]

                ax[ii].plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
                cntr = ax[ii].tricontourf(self.triang_b2, np.log10(self.quants[field]).flatten(),
                                           cmap=cm.magma,levels=300)

                # create draggable colorbar
                cbars.append( plt.colorbar(cntr, format='%.3g', ax=ax[ii]))
                cbars[-1].ax.set_title(label)
                cbars[-1] = plot_tools.DraggableColorbar(cbars[-1],cntr)
                cbars[-1].connect()

                ax[ii].axis('equal')
                ax[ii].set_xlabel('R [m]')
                ax[ii].set_ylabel('Z [m]')



    def plot_2d_vals(self, vals, label='', use_triang=True, max_mask_len=None):
        '''Method to plot 2D fields over the R and Z grids. 
        Colorbars are set to be manually adjustable, allowing variable image saturation.

        Args:
            vals : array, (ny,nx)
                2D array containing a variable of interest, on the same grid as the 
                R and Z attributes. 

        Keyword Args:
            label : str
                Label describing the quantity being plotted.
            use_triang : bool
                If True, use a triangulation to plot results on B2 grids, attempting
                to mask unwanted spurious edges. Alternatively, use a simple filled 
                contour plot, which however may show unphysical features at the edges
                of the B2 grid. Default is True (use the triangulation).  
            max_mask_len : float
                Optional maximum length of triangulation segments to be masked. If left to 
                None, the default rule-of-thumb length for this variable will be used, but 
                this may need hand adjustments for specific use cases.

        '''

        fig,ax0 = plt.subplots(figsize=(8,11))
        ax0.axis('equal')
        ax0.set_xlabel('R [m]'); ax0.set_ylabel('Z [m]')

        # plot LCFS
        ax0.plot(self.geqdsk['RBBBS'],self.geqdsk['ZBBBS'],c='k')
        
        if use_triang:
            # possibly modify edge masking of triangulation
            if max_mask_len is not None:
                self.max_mask_len = float(max_mask_len)
                self.triang_b2 = apply_mask(self.triang_b2, self.geqdsk, max_mask_len=max_mask_len)
            
            # triangulation from b2 R and Z grid
            cntr = ax0.tricontourf(self.triang_b2, vals.flatten(), cmap=cm.magma,levels=300)
        else:
            # Simpler plotting, with no triangulation
            cntr = ax0.contourf(self.R, self.Z, vals)

        cbar = plt.colorbar(cntr, format='%.3g', ax=ax0)
        cbar.set_label(label)
        cbar = plot_tools.DraggableColorbar(cbar,cntr)
        cbar.connect()


    def get_radial_prof(self, quant='nn', dz_mm=5, plot=False):
        '''Extract radial profiles of a quantity "quant" from the SOLPS run. 
        This function returns profiles on the low- (LFS) and high-field-side (HFS) midplane, 
        as well as flux surface averaged (FSA) ones. 

        Keyword Args:
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

        Returns:
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
                                (r,z), method='cubic')

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
                             method='cubic')
        _prof_LFS[_prof_LFS<0]=np.nan
        R_LFS = np.linspace(np.min(R_midplane_lfs), np.max(R_midplane_lfs),100)
        rhop_LFS = coords.get_rhop_RZ(R_LFS,np.zeros_like(R_LFS), self.geqdsk)

        # ... and on the HFS:
        _prof_HFS = griddata((self.R.flatten(),self.Z.flatten()), self.quants[quant].flatten(),
                             #(_R_HFS, np.zeros_like(_R_HFS)),
                             (_R_HFS,0.5*dz_mm*1e-3*np.random.random(len(_R_HFS))),
                             method='cubic')
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
        HFS_mid_pol_idx = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]<self.geqdsk['RMAXIS']][midplane_HFS_idx]
        LFS_mid_pol_idx = self.unit_p + np.arange(Z_core.shape[1])[R_core[0,:]>self.geqdsk['RMAXIS']][midplane_LFS_idx]

        # find rhop along midplane grid chords
        rhop_chord_HFS = coords.get_rhop_RZ(self.R[:,HFS_mid_pol_idx],self.Z[:,HFS_mid_pol_idx], self.geqdsk)
        rhop_chord_LFS = coords.get_rhop_RZ(self.R[:,LFS_mid_pol_idx],self.Z[:,LFS_mid_pol_idx], self.geqdsk)

        if plot:
            # plot quantties on linear scale -- need to adapt labels
            lab = fr'${self.labels[quant].split("$")[1]}$ [${self.labels[quant].split("$")[3]}$]'
            
            # compare FSA radial profiles with midplane (LFS and HFS) ones
            fig,ax = plt.subplots()
            ax.plot(rhop_FSA, prof_FSA, label='FSA')
            ax.plot(rhop_LFS, prof_LFS, label='LFS midplane')
            ax.plot(rhop_HFS, prof_HFS, label='HFS midplane')
            ax.plot(rhop_chord_LFS, self.quants[quant][:,LFS_mid_pol_idx], label='LFS grid midplane')
            ax.plot(rhop_chord_HFS, self.quants[quant][:,HFS_mid_pol_idx], label='HFS grid midplane')
            ax.set_xlabel(r'$\rho_p$')
            ax.set_ylabel(lab)
            ax.legend(loc='best').set_draggable(True)
            plt.tight_layout()

        return rhop_FSA, prof_FSA, rhop_LFS, prof_LFS, rhop_HFS, prof_HFS


def apply_mask(triang, geqdsk, max_mask_len=0.4, mask_up=False, mask_down=False):
    '''Function to apply basic masking to a matplolib triangulation. This type of masking
    is useful to avoid having triangulation edges going outside of the true simulation
    grid. 

    Args:
        triang : instance of matplotlib.tri.triangulation.Triangulation
            Matplotlib triangulation object for the (R,Z) grid. 
        geqdsk : dict
            Dictionary containing gEQDSK file values as processed by the `omfit_eqdsk`
            package. 

    Keyword Args:
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
      
    Returns:
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
