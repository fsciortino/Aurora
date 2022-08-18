'''Post-processing tools for OEDGE.
Adapted and extended based on original work by S.Zamperini, J.Nichols and D.Elder.
'''
import sys, re, os
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import shutil
from collections import OrderedDict
from scipy import constants

from scipy.interpolate import griddata, interp1d

from . import radiation

class oedge_input(dict):
    """Methods to load and process OEDGE input data files.
    Allows modification and re-writing of files for new OEDGE runs.
    """

    def __init__(self, filename):
        self.filename = filename

        # load chosen input file
        self.load()

        # dictionary of input tag categories
        self.category_dict = {'General': ['088', '089', '020', 'S11', 'S12', 'S13',
                                          'S14', 'I15', 'T14', 'S02', 'S03', 'S04',
                                          'S05', 'S15', 'S21', 'H03', 'A01', 'A02',
                                          'A03', 'A04', 'Q32', 'Q33', 'Q34', 'Q35',
                                          'Q36', 'Q37', 'Q44', 'P36'],
                              'Grid': ['G'],
                              'DIVIMP': ['I'],
                              'Neutrals': ['N'],
                              'Plasma': ['P', 'R', 'Q'],
                              'SOL22': ['2'],
                              'SOL23': ['3'],
                              'OSM': ['{'],
                              'EIR': ['0', 'E'],
                              'Fluid code': ['F'],
                              'Data': ['D'],
                              'Physics': ['T', 'S'],
                              'Diags': ['C'],
                              'H/HC': ['H'],
                              'Other': ['B']}
        
    def __repr__(self):
        '''Display loaded input file in nice form.
        NB: one can also print part of the output via a call like
        >> print(inp.__repr__()[:2000])
        '''
        return '\n'.join(self.create_input())
    
    def load(self):
        '''Load an OEDGE input file. '''
        with open(self.filename, 'r') as f:
            lines = f.read()

        # handle inline comments with $
        lines = re.sub(r'(?<!\n)\$', '\n$', lines) # adds \n before $
        lines = lines.split('\n')

        # split sections
        c = 0
        inhibit = False
        for line0 in lines:
            line = line0.rstrip()

            # stop parsing when find a line starting with c
            if line.startswith('c'):
                inhibit = True
                self['__comment_%06d__' % c] = line0
                c += 1

            # new entry with Z'{ style
            elif not inhibit and line.startswith("'{"):
                item = re.sub(r'.*(\{.*\}).*', r'\1', line.split("'")[1])
                self[item] = OrderedDict()
                self[item]['__raw__'] = [line]
                
            # new entry with '* or '+ style
            elif not inhibit and re.match(r'^\'[\*\+][a-zA-Z0-9][0-9][0-9] ', line):
                item = line.split("'")[1][:4]
                # Item C14 is repeated twice in some input files - the second should be C15
                if item in self:
                    # found a repeated item
                    # if the repeat item is C14 replace with C15
                    if item == '+C14':
                        item == '+C15'
                self[item] = OrderedDict()
                self[item]['__raw__'] = [line]

            # continuation of entry
            elif not inhibit and re.match(r'^ *[\'0-9\.\-].*', line):
                # Think this is where the other mislabeled lines go
                # Kludgy fix to a bug in older input files
                #
                if 'Ring Range :-' in line:
                    # insert tag at beginning of line
                    item = '+F07'
                    line = "'" + item + line[1:]
                    self[item] = OrderedDict()
                    self[item]['__raw__'] = [line]
                elif 'J0 & J1    :-' in line:
                    item = '+F08'
                    line = "'" + item + line[1:]
                    self[item] = OrderedDict()
                    self[item]['__raw__'] = [line]
                elif 'VMF0,1,2   :-' in line:
                    item = '+F09'
                    line = "'" + item + line[1:]
                    self[item] = OrderedDict()
                    self[item]['__raw__'] = [line]
                else:
                    self[item]['__raw__'].append(line)
            # comment
            else:
                self['__comment_%06d__' % c] = line0
                c += 1

        # interpret sections
        for item in list(self.keys()):
            if item.startswith('__comment_'):
                continue

            # initialize the tag field for all non-comment input items
            self[item]['tag'] = ''

            raw = self[item]['__raw__']
            raw0 = raw[0].strip().split('\'')
            # special {...}
            if '{' in item:
                self[item]['comment'] = raw0[1].split('}')[1].strip()
                self[item]['data'] = []
                self[item]['tag'] = item
                self[item]['inline_comment'] = ''

                try:
                    dum = int(raw0[2]) # if this works, then scalar
                    self[item]['data'], inline_comment = interpret(raw0[2])
                    # scalars as scalars
                    if len(self[item]['data']) == 1:
                        self[item]['data'] = self[item]['data'][0]

                except Exception:
                    # array
                    for line in raw[1:]:
                        data, inline_comment = interpret(line)
                        self[item]['data'].append(data)
            # standard
            else:
                self[item]['comment'] = raw0[1][5:].strip()
                self[item]['tag'] = item[1:]

                # scalar(ish)
                if len(raw) == 1:
                    tmp = interpret(raw[0])
                    self[item]['data'] = tmp[0][1:]
                    self[item]['inline_comment'] = tmp[-1]
                    # scalars as scalars
                    if len(self[item]['data']) == 1:
                        self[item]['data'] = self[item]['data'][0]

                # array
                else:
                    self[item]['inline_comment'] = '\''.join(raw0[2:])
                    self[item]['inline_comment'] = self[item]['inline_comment'].strip(' \'')
                    for ndummies, line in enumerate(raw[1:]):
                        if "' '" not in line:
                            break

                    header = re.sub(r'(\'.*\'\s*)([0-9]+)(\s.*)', r'\1',
                                    raw[1 + ndummies] + ' ').strip('\' ')
                    try:
                        n = int(re.sub(r'(\'.*\'\s*)([0-9]+)(\s.*)', r'\2',
                                       raw[1 + ndummies] + ' ').strip('\' '))
                    except:
                        # added by FS to deal with some AUG cases -- revise!
                        n = 0
                    header_comment = re.sub(r'(\'.*\'\s*)([0-9]+)(\s.*)', r'\3',
                                            raw[1 + ndummies] + ' ').strip()
                    data = []
                    for k in range(len(raw[2 + ndummies :])):
                        data.append(interpret(raw[2 + ndummies + k])[0])
                    dummies = raw[1 : ndummies + 1]
                    self[item]['dummies'] = dummies
                    self[item]['header'] = header
                    self[item]['n'] = n
                    self[item]['header_comment'] = header_comment
                    self[item]['data'] = data

    def get_category(self, tag):
        let = tag[0]
        category = None
        if tag in category_dict['General']:
            category = 'General'
        else:
            for item in category_dict:
                if let in category_dict[item]:
                    category = item
                    break
        if category is None:
            category = 'OTHER'
        return category

    def create_input(self):
        '''Set line lengths to give a more nicely formatted output file that is more legible form.
        '''
        tag_width = 4
        comment_width = 50
        end_width = 20
        data_width = 10

        # import here to avoid issues during regression tests
        from omfit_classes.utils_base import tolist
        
        # create output list
        out = []
        for item in self:
            if item.startswith('__comment_'):
                out.append(self[item].strip().ljust(comment_width))  #FS
                #pass

            elif '{' in item:
                out.append('\'%s %s\'' % (item.ljust(tag_width), self[item]['comment'].strip().ljust(comment_width))) #FS
                #out.append('\'%s \'' %item.ljust(tag_width))
                if isinstance(self[item]['data'], list):
                    for line in self[item]['data']:
                        out.append(' '.join(item.rjust(data_width) for item in map(repr, line)))
                else:
                    out[-1] += ' ' + str(self[item]['data']).strip().rjust(data_width)

            elif 'header_comment' in self[item]:
                if self[item]['comment'] == '':
                    out.append( 
                        '\'%s %s\' \'%s\'' % (item.ljust(tag_width), self[item]['comment'].strip(), self[item]['inline_comment'].strip())
                    )   # FS
                    for dummy in self[item]['dummies']:
                        out.append(dummy.strip().ljust(comment_width))
                    out.append(
                        '\'%s\' %s %s'
                        % (
                            self[item]['header'].strip().ljust(comment_width),
                            str(len(self[item]['data'])).rjust(data_width),
                            self[item]['header_comment'].rjust(end_width),
                        )
                    )
                    for line in self[item]['data']:
                        out.append(' '.join(item.rjust(data_width) for item in map(repr, line)))
                else:
                    out.append(
                        '\'%s %s\'  %s ' % (item.ljust(tag_width), self[item]['comment'].strip(), self[item]['inline_comment'].strip())
                    ) #FS
                    #out.append('\'%s \'' % item.ljust(tag_width))
                
                    for dummy in self[item]['dummies']:
                        out.append(dummy.strip().ljust(comment_width))
                    out.append(
                        '\'%s\' %s %s'
                        % (
                            self[item]['header'].strip().ljust(comment_width),
                            str(len(self[item]['data'])).rjust(data_width),
                            self[item]['header_comment'].rjust(end_width),
                        )
                    )
                    for line in self[item]['data']:
                        out.append(' '.join(item.rjust(data_width) for item in map(repr, line)))

            elif 'inline_comment' in self[item]:
                if self[item]['comment'] == '':
                    out.append(
                        '\'%s %s\' %s %s'
                        % (
                            item.ljust(tag_width),
                            self[item]['comment'].strip(),
                            ' '.join(item.rjust(data_width) for item in map(repr, tolist(self[item]['data']))),
                            self[item]['inline_comment'],
                        )
                    )
                elif isinstance(self[item]['data'], str):
                    out.append(
                        '\'%s %s\' %s %s'
                        % (
                            item.ljust(tag_width),
                            self[item]['comment'].strip(),
                            ' '.join(item.rjust(data_width) for item in map(repr, tolist(self[item]['data']))),
                            self[item]['inline_comment'],
                        )
                    )
                else:
                    out.append(
                        '\'%s %s\' %s %s'
                        % (
                            item.ljust(tag_width),
                            self[item]['comment'].strip().ljust(comment_width),
                            ' '.join(item.rjust(data_width) for item in map(repr, tolist(self[item]['data']))),
                            self[item]['inline_comment'].rjust(end_width),
                        )
                    )

        return out

    def create_input_simple(self):
        '''Similar to `create_input`, but exclude all comments and just give lines providing real inputs.
        '''        
        out = []
        for ii, key in enumerate(self.keys()):
            
            if isinstance(self[key], str): # just comment
                continue

            out.append(self[key]['__raw__'][0])

        return out
        
    def write_input_file(self, filename=None):
        '''Write the OEDGE options currently in `self` to a file of given name.

        Users can modify input options by changing the loaded inputs before writing a new input file, e.g.
        >> oedgein = oedge_input(myfile)
        >> oedgein['+228']['data'] = [0000, 1111]
        >> oedgein.write_input_file(filename='mynewfile.d6i')
        '''
        new_input = self.create_input()

        with open(filename, 'w') as f:
            f.write('\n'.join(new_input))


class oedge_case(dict):
    '''Class to orchestrate OEDGE runs, loading of results, and postprocessing.

    Automatic grid creation routines are currently available for AUG and TCV.
    For analysis of other devices, grids should be created externally, 
    e.g. via DivGeo/Carre.

    Expected workflow:
    - instantiate class object
    - load an input file
    - modify the input options file as needed
    - create a grid file, if needed
    - run OEDGE (input options and grid files are needed)
    - load output
    - plot/postprocess results

    In practice, after the class initialization, users may skip any of these steps
    as convenient.
    '''
    def __init__(self, shot, time_s, device='AUG', base=None, label=''):
        '''OEDGE case initialization. Shot number and time are used as identifiers
        of the OEDGE run. If automatic grid generation is invoked, these inputs are
        also used to load a magnetic equilibrium and create 2D grid files.

        Parameters
        -----------
        shot: int
            AUG shot number.
        time_s: float
            Shot time [s] of interest
        device: str
            Device name, e.g. "AUG". This can in principle be used as a label,
            assigning any string to it.
        base : str
            Directory where inputs and outputs should be located. If None, 
            this is set to a default "divimp" directory in the user home. Default is None.
        label: str
            Additional label describing the case, e.g. "osm" or "divimp" or "newcase"
        '''
        self.shot = int(shot)
        self.time_s = float(time_s)
        self.device = str(device)
        self.label = str(label)
        
        if base is None:
            # assumed location of divimp directory in user's home
            self.base = os.path.expanduser('~')+'/divimp'
        else:
            self.base = str(base)
            
        # extra label for specific case
        extra = f'_{label}' if len(label) else ''

        # files nomenclature (does not include file extension/type)
        self.files_basename = f'oedge_{self.device}_{self.shot}_{int(self.time_s*1e3)}{extra}'
        
        # default file nomenclatures
        self.input_filepath = self.base+f'/data/{self.files_basename}.d6i'
        self.grid_filepath = self.base+f'/shots/{self.files_basename}.sno'
        self.output_nc_file = self.base+f'/results/{self.files_basename}.nc'

    def __repr__(self):
        return f'OEDGE case for shot {self.shot}, time {self.time_s}s'

    def load_input_file(self, filepath=None):
        '''Load and process OEDGE input file (.d6i format).

        Parameters
        ----------
        filepath: str or None
            Input filename (in .d6i format) to be loaded.
            This will be saved in the file 'oedge_{device}_{shot}_{time in ms}.d6i'. 
            NOTE: If this already exists, it will be overwritten!
        '''        
        self.inputs = oedge_input(filepath)

        # modify equilibrium in input file to match the one of current case
        self.inputs['+A03']['data'] = self.grid_filepath.split('/')[-1]
        
        # now save under a new filename for the current OEDGE case
        self.inputs.write_input_file(self.input_filepath)

    def run(self, grid_loc=None, ghost_filename=None, submit=True):
        '''Launch a OEDGE simulation.

        Parameters
        ----------
        grid_loc : str
            Path to the grid file. If left to None, a default path and nomenclature
            to the grid file is assumed for the current simulation.
        ghost_filename : str
            Path to a background "ghost" file for using with DIVIMP.
            If left to None, no ghost file is given as input and (depending on inputs)
            OSM-EIRENE may be run.
        submit : bool
            If True, the run will be submitted to the current server.
            If False, the run command will be returned without submitting the job.
            The latter option allows one to more easily use the run command within 
            a job control manager like SLURM, if desirable.
        '''
        if not os.path.exists(self.input_filepath):
            raise ValueError('Select an existing input file using the `load_input_file` method')
        if grid_loc is not None and os.path.exists(grid_loc):
            self.grid_filepath = grid_loc
        if not os.path.exists(self.grid_filepath):
            raise ValueError('Select a grid file first using the `load_grid_file` method')

        # update case name in input file
        self.inputs['+A01']['data'] = f'{self.device}, shot {self.shot}, time {self.time_s}'
        self.inputs.write_input_file(self.input_filepath)
        
        # NB: when running with rd, location of files is assumed; only file names are needed
        input_filename = self.input_filepath.split('/')[-1]
        input_grid_filename = self.grid_filepath.split('/')[-1]
        
        # ignore the OUT input file, assuming it is disabled
        cwd = os.getcwd()
        os.chdir(self.base)
        c1 = f' {input_filename} dummy {input_grid_filename}'
        if ghost_filename is not None:
            self.ghost_filename = ' '+ str(ghost_filename)
            c2 = ' -o' + self.ghost_filename
        else:
            c2 = ''
        run_command = './rd' +c2 + c1
        if submit:
            print(f'Running: \n {run_command}')
            os.system(run_command)
            os.chdir(cwd)
        else:
            os.chdir(cwd)
            return run_command
        
    def load_output(self):
        '''Load output netCDF file and allow plotting/postprocessing
        '''
        self.output = oedge_output(self.output_nc_file)

class oedge_output:
    def __init__(self, output_nc_file=None):
        """Loads in the OEDGE output from a netCDF file.

        Parameters
        ----------
        output_nc_file: str or None
            Path location of .nc output file. If not provided, attempts to load
            a file of default nomenclature 'oedge_{device}_{shot}_{time in ms}.nc'.
        """
        if output_nc_file is not None:
            self.output_nc_file = str(output_nc_file)

        # import omfit_classes here to prevent import during regression tests
        from omfit_classes.omfit_nc import OMFITnc
        
        # Load in the output netCDF file
        self.nc = OMFITnc(self.output_nc_file)

        # Load in some netCDF data that is used a lot.
        self.rs     = self.nc['RS']['data'] # R coordinante of cell centers
        self.zs     = self.nc['ZS']['data'] # Z coordinate of cell centers
        self.nrs    = int(self.nc['NRS']['data']) # Number of rings on the grid
        self.nks    = self.nc['NKS']['data']  # Number of knots on each ring
        self.nds    = self.nc['MAXNDS']['data'] # Maximum Number of target elements
        self.area   = self.nc['KAREAS']['data']  # cell area
        self.korpg  = self.nc['KORPG']['data']  # IK,IR mapping to polygon index
        self.rvertp = self.nc['RVERTP']['data'] # R corner coordinates of grid polygons
        self.zvertp = self.nc['ZVERTP']['data'] # Z corner coordinates of grid polygons
        self.rvesm  = self.nc['RVESM']['data'] # R coordinates of vessel wall segment end points
        self.zvesm  = self.nc['ZVESM']['data'] # V coordinates of vessel wall segment end points

        # indexes for rings defining the SOL rings
        self.irsep  = int(self.nc['IRSEP']['data']) # Index of first ring in main SOL
        self.irwall = int(self.nc['IRWALL']['data']) # Index of outermost ring in main SOL
        self.irwall2 = int(self.nc['IRWALL2']['data']) # Second wall ring in double null grid
        self.irtrap = int(self.nc['IRTRAP']['data']) # Index of outermost ring in PFZ
        self.irtrap2 = int(self.nc['IRTRAP2']['data']) # Index of outermost ring in second PFZ
        
        # other
        self.qtim   = self.nc['QTIM']['data']  # Time step for ions
        self.kss    = self.nc['KSS']['data'] # S coordinate of cell centers along the field lines
        self.kfizs  = self.nc['KFIZS']['data']  # Impurity ionization rate
        self.ksmaxs = self.nc['KSMAXS']['data'] # S max value for each ring (connection length)
        
        self.crmb   = self.nc['CRMB']['data'] # Mass of plasma species in amu
        self.crmi   = self.nc['CRMI']['data'] # Mass of impurity species in amu
        self.cion   = self.nc['CION']['data'] # Atomic number of impurity species

        # eliminate spaces from some netcdf keys
        for field in ['ZC IN','ZC OUT','RC IN','RC OUT']:
            self.nc[field.replace(' ','_')] = self.nc[field]
            del self.nc[field]

        # estimate neutral pressure that would be measured by an ASDEX gauge
        try:
            Twall = 300 * constants.k/constants.e # 300K in eV
            n_gauge = (self.nc['PINATO']['data']/np.sqrt(2))*np.sqrt(self.nc['PINENA']['data']/Twall) +\
                      self.nc['PINMOL']['data']*np.sqrt(self.nc['PINENM']['data']/Twall)
            self.nc['p0'] = {'data': n_gauge * Twall * constants.k} # Pa = N/m^2
        except:
            # EIRENE results not necessarily loaded in DIVIMP runs
            pass
        
        # load convenient descriptions of variables in `varnames` attribute
        self.load_var_descriptions()
        
        # All impurity results are stored scaled to 1 part/m-tor/s entering the system
        # to get absolute values the results are multipled by absfac (absolute scaling factor)
        self.absfac = self.nc['ABSFAC']['data'] if 'ABSFAC' in self.nc else 1.

        # Simulation (ion) time steps
        self.qtim = self.nc['QTIM']['data'] if 'QTIM' in self.nc else 1.

        # neutral time step
        self.fsrate = self.nc['FSRATE']['data'] if 'FSRATE' in self.nc else 1.

        # number of impurity charge states
        self.nizs = self.nc['NIZS']['data'] if 'NIZS' in self.nc else None

        # Create a mesh of of the corners of the each cell/polygon in the grid.
        self.mesh = []
        self.mesh_idxs = []
        num_cells = 0
        
        # Scan through the rings:
        for ir in range(self.nrs):

            # Scan through the knots on each ring:
            for ik in range(self.nks[ir]):

                # Get the cell index of this knot on this ring
                index = self.korpg[ir,ik] - 1

                # Only if the area of this cell is not zero append the corners
                if self.area[ir,ik] != 0.0:
                    vertices = list(zip(self.rvertp[index][:4], self.zvertp[index][:4]))
                    self.mesh.append(vertices)
                    num_cells = num_cells + 1

                    # Print out a warning if the cell center is not within the vertices
                    cell = mpl.path.Path(list(vertices))

                    r = self.rs[ir, ik]
                    z = self.zs[ir, ik]
                    if not cell.contains_point([r, z]):
                        print("Error: Cell center not within vertices.")
                        print("  (ir, ik)    = ({}, {})".format(ir, ik))
                        print("  Vertices    = {}".format(vertices))
                        print("  Cell center = ({}, {})".format(r, z))
                    else:
                        self.mesh_idxs.append([ir,ik])
        self.num_cells = num_cells

        # read in also the .dat file, containing info on the input options/data
        dat_path = output_nc_file.split(".nc")[0] + ".dat"
        if os.path.exists(dat_path):
            with open(dat_path) as f:
                self.dat_file = f.read()

            try:
                # read total run time -- seems robust to DIVIMP versions
                self.cpu_time = float(self.dat_file.split('CPU TIME USED')[-1].split('(S)')[-1].split('\n')[0])
            except:
                print('Could not read CPU time from .dat file')
                self.cpu_time = np.nan
        else:
            self.dat_file = None
            print('Could not automatically read the .dat file')

    def __repr__(self):
        '''Briefly describe loaded case on the command line.
            '''
        message = '  OEDGE case:  ' + str(self.nc['TITLE']['data']) + '\n' +\
                  '  Date run:    ' + str(self.nc['JOB']['data']).strip()[:8] + '\n' +\
                  '  Grid:        ' + str(self.nc['EQUIL']['data']) + '\n' +\
                  '  Description: ' + str(self.nc['DESC']['data']) + '\n'
        return message
            
    @property
    def name_maps(self):
        '''Create a dictionary showing correspondences of output variable
        names and more intuitive nomenclature (e.g. "ne", "Te", etc.), as
        well as variable labels and units.
        '''
        return {'ne': {'data': 'KNBS', 'targ': 'KNDS', 'label': r'$n_e$', 'units': r'$m^{-3}$'},
                'Te': {'data': 'KTEBS', 'targ': 'KTEDS', 'label': r'$T_e$', 'units': 'eV'},
                'Ti': {'data': 'KTIBS', 'targ': 'KTIDS', 'label': 'Ti', 'units': 'eV'},
                'Vb': {'data': 'KVHS', 'targ': 'KVDS', 'label': r'$v_{||}$', 'units': 'm/s', 'scale': '1/QTIM'},
                'Epar': {'data': 'KES', 'targ': 'KEDS', 'label': r'$E_{||}$', 'units': 'V/m'},
                'ExB_pol': {'data': 'E_POL', 'targ': None, 'label': r'$E_\theta$', 'units': 'V/m'},
                'ExB_rad': {'data': 'E_RAD', 'targ': None, 'label': r'$E_r$', 'units': 'V/m'},
                'V_pol': {'data': 'EXB_P', 'targ': None, 'label': 'Vpol', 'units': 'm/s'},
                'V_rad': {'data': 'EXB_R', 'targ': None, 'label': 'Vrad', 'units': 'm/s'},
                # only for cases that run EIRENE:
                'H Prad': {'data': 'HPOWLS', 'targ': None, 'label': r'$P_{D,rad}$', 'units': r'$W/m^3$'},
                'H Pline': {'data': 'HLINES', 'targ': None, 'label': r'$P_{D,line}$', 'units': r'$W/m^3$'},
                'H Dalpha': {'data': 'PINALP', 'targ': None, 'label': r'$P_{D-\alpha}$', 'units': r'$ph/m^3/s$'},
                'H ioniz': {'data': 'PINION', 'targ': None, 'label': r'$S_D$', 'units': r'$1/m^3/s$'},
                'H recomb': {'data': 'PINREC', 'targ': None, 'label': r'$R_D$', 'units': r'$1/m^3/s$'},
                'n_H2': {'data': 'PINMOL', 'targ': None, 'label': r'$n_{D,mol}$', 'units': r'$1/m^3$'},
                'n_H': {'data': 'PINATO', 'targ': None, 'label': r'$n_{D,n}$', 'units': r'$1/m^3$'},
                'T_H': {'data': 'PINENA', 'targ': None, 'label': r'$T_{D,n}$', 'units': 'eV'},
                'T_H2': {'data': 'PINENM', 'targ': None, 'label': r'$T_{D,mol}$', 'units': 'eV'},
                'H ion energy loss': {'data': 'PINQI', 'targ': None,
                                      'label': 'Hydrogen-Ion Energy Loss Term',
                                      'units': r'$W/m^3$'}, # to check
                'H elec energy loss': {'data': 'PINQE', 'targ': None,
                                       'label': 'Hydrogen-Electron Energy Loss Term',
                                       'units': r'$W/m^3$'}, # to check
                # impurity-related quantities (only if DIVIMP was run):
                'nimp': {'data': 'DDLIMS', 'targ': None, 'label': r'$n_z$', 'units': r'$1/m^3$', 'scale': 'ABSFAC'},
                'Timp': {'data': 'DDTS', 'targ': None, 'label': r'$T_z$', 'units': 'eV'},
                'S_z': {'data': 'TIZS', 'targ': None, # impurity ionization
                        'label': r'S_z$', 'units': r'$1/m^3/s$', 'scale': 'ABSFAC'},
                'imp Prad': {'data': 'POWLS', 'targ': None, 'label': r'$P_z$',
                             'units': r'$W/m^3$', 'scale': 'ABSFAC'},
                # added by this class
                'p0': {'data': 'p0', 'label': r'$p_0$', 'units': r'$N/m^2$'}
        }


    def read_data_2d(self, dataname, charge=None, scaling=1.0, no_core=False):
        """
        Reads in 2D data into a 1D array, in a form that is then passed easily
        to PolyCollection for plotting.

        Parameters
        ----------
        dataname : str
            The 2D data as named in the netCDF file.
        charge : int
            The charge state to be plotted, if applicable.
        scaling : Scaling factor to apply to the data, if applicable. Secret
            option is 'Ring' to just return the ring number at each cell.
        no_core: bool
            If True, exclude any core data.

        Returns
        -------
        data : The data in a 2D format compatible with plot_2d.
        """
        # Normalized s coordinate of each cell.
        if dataname == 'Snorm':
            raw_data = self.nc['KSB']['data'] / (self.ksmaxs[:, None]+1e-30)
            raw_data = np.abs(raw_data + raw_data / 1.0 - 1.0)
        elif isinstance(dataname, str):
            raw_data = self.nc[dataname]['data']
        elif dataname.shape==self.nc['KNBS']['data'].shape:
            # user provided data in 2D format of nc dictionary data, to be converted for contour plotting.            
            raw_data = copy.deepcopy(dataname)
            dataname = 'dum'
        else:
            raise ValueError('Unrecognized input to read_data_2d method!')

        data = np.zeros(self.num_cells)

        if charge in [None, 'all'] and dataname.lower() in ["ddlims",'powls','ddts','tizs']:
            # sum over all charge states
            # The first entry are "primary" neutrals, and the second are "total" neutrals. 
            # So, we only want to include the second entry onwards, i.e.
            # 0 - C0primary, 1 - C0total, 2 - C1+, ... 7 - C6+.
            raw_data = raw_data[1:].sum(axis=0)

        count = 0
        for ir in range(self.nrs):
            for ik in range(self.nks[ir]):
                if self.area[ir, ik] != 0.0:

                    # label data by ring number
                    if scaling == 'Ring':
                        data[count] = ir + 1 # correct indexing begins at 1
                    elif scaling == 'Knot':
                        data[count] = ik + 1 # correct indexing begins at 1
                    else:
                        # If charge is specified, this will be the first dimension.
                        # NB: raw_data was stored above such that charge=0 is for neutrals, charge=1 is for Z1+, etc.
                        if charge in [None, 'all']:
                            data[count] = raw_data[ir][ik] * scaling
                        else:
                            data[count] = raw_data[charge][ir][ik] * scaling

                        if no_core: # replace core data with small number
                            if ir < self.irsep - 1:
                                data[count] = None #sys.float_info.epsilon
                    count = count + 1

        # prevent large numbers (e.g. 9.99e36) from messing up figures
        data[np.where(data > 1e30)] = None

        return data

    def read_data_2d_kvhs_t13(self, no_core=False):
        """
        Special function for plotting the flow velocity. This
        is because some DIVIMP options (T13, T31, T37?, T38?...) add
        additional flow values not reflected in KVHS. These additional values
        are in the .dat file, and thus it is required to run this function.

        Input
        no_core : Exclude the core data.

        Output
        data : The data in a 2D format compatible with plot_2d.
        """

        # Make sure .dat file has been loaded.
        if self.dat_file is None:
            raise ValueError(".dat file not loaded in! Run 'add_dat_file' first.")

        try:
            pol_opt = float(self.dat_file.split('POL DRIFT OPT')[1].split(':')[0])
            if pol_opt == 0.0:
                print("Error: Poloidal drift option T13 was not on for this run.")
                return -1

            # Get the relevant table for the extra drifts out of the .dat file.
            add_data = self.dat_file.split('TABLE OF DRIFT REGION BY RING - RINGS ' + \
                                            'WITHOUT FLOW ARE NOT LISTED\n')[1]. \
                                            split('DRIFT')[0].split('\n')

            # Split the data between the spaces, put into DataFrame.
            add_data = [line.split() for line in add_data]
            add_df = pd.DataFrame(add_data[1:-1], columns=['IR', 'Vdrift (m/s)',
                                  'S_START (m)', 'S_END (m)'], dtype=float). \
                                  set_index('IR')

            # Get the 2D data from the netCDF file.
            dataname = 'KVHS'
            scaling = 1.0 / self.qtim
            raw_data = self.nc[dataname]['data']
            data = np.zeros(self.num_cells)

            # Convert the 2D data (ir, ik) into 1D for plotting in the PolyCollection
            # matplotlib function.
            count = 0
            for ir in range(self.nrs):
                for ik in range(self.nks[ir]):
                    if self.area[ir, ik] != 0.0:

                        # Put the data from this [ring, knot] into a 1D array.
                        data[count] = raw_data[ir][ik] * scaling

                        # If this ring has additional drifts to be added.
                        if ir in add_df.index:

                            # Then add the drift along the appropriate s (or knot) range.
                            if self.kss[ir][ik] > add_df['S_START (m)'].loc[ir] and \
                               self.kss[ir][ik] < add_df['S_END (m)'].loc[ir]:

                               data[count] = data[count] + add_df['Vdrift (m/s)'].loc[ir]

                        if no_core:  # exclude data from the core
                            if ir < self.irsep - 1:
                                data[count] = sys.float_info.epsilon

                        count = count + 1

            return data

        except IndexError:
            print('Error: Was T13 on for this run?')
            return None

    def get_sep(self):
        """
        Return collection of lines to be plotted with LineCollection method
        of matplotlib for the separatrix.

        Returns
        -------
        lines : List of coordinates to draw the separatrix in a format friendly
                 for LineCollection.
        """

        # Get (R, Z) coordinates of separatrix.
        rsep = self.rvertp[self.korpg[self.irsep-1,:self.nks[self.irsep-1]]][:,0]
        zsep = self.zvertp[self.korpg[self.irsep-1,:self.nks[self.irsep-1]]][:,0]
        nsep = len(rsep)
        lines=[]

        # Construct separatrix as a series of pairs of coordinates (i.e. a line
        # between the coordinates), to be plotted. Don't connect final point to first.
        for i in range(nsep-2):
            lines.append([(rsep[i], zsep[i]), (rsep[i+1], zsep[i+1])])

        return lines

    def calculate_forces(self, force, charge, vz_mult = 0.0, no_core=False):
        """
        Return 2D representations of the parallel forces (FF, FiG, FeG, FPG, FE)
        in the same format as read_data_2d for easy plotting. This data is not
        returned in the NetCDF file, so we calculate it here. Ideally we would
        use exactly what DIVIMP calculates, but that doesn't seem to be here yet.

        Parameters
        ----------
        force : str
            One of 'FF', 'FiG', 'FeG', 'FPG', 'FE' or 'Fnet' (to sum them all).
        charge : None or int
            Charge of impurity ion, needed for some of the forces.
        vz_mult : float
            Fraction of the background velocity used for vz (needed only in FF).
        no_core : bool
            If True, exclude core data.

        Returns
        -------
        force : 2D array
            The force data in a 2D format compatible with 
            ~aurora.oedge.oedge_case.plot_2d.
        """
        # Temperature and density values for calculations
        te = self.read_data_2d('KTEBS', no_core=no_core)
        ti = self.read_data_2d('KTIBS', no_core=no_core)
        ne = self.read_data_2d('KNBS',  no_core=no_core)
        col_log = 15
        fact = np.power(self.qtim, 2) * constants.e / (self.crmi * constants.m_p)

        # See if T13 was on. Important for FF calculations
        try:
            pol_opt = float(self.dat_file.split('POL DRIFT OPT')[1].split(':')[0])
            if pol_opt == 0.0:
                t13 = False
            elif pol_opt == 1.0:
                print("Additional poloidal drift option T13 was ON.")
                t13 = True

        except:
            print("Warning: Unable to determine if T13 was on/off.")
            print("Was .dat file loaded?")

        if force.lower() in ['ff', 'fnet']:  # Friction force

            try:
                # Spitzer slowing down time, Stangeby p.317, eq. 6.35
                tau_s = 1.47e13 * self.crmi * ti * np.sqrt(ti / self.crmb) / \
                        ((1 + self.crmb / self.crmi) * ne * np.power(charge, 2) * col_log)

            except TypeError:
                raise ValueError("Charge of ion needed for friction force calculation!")

            # TODO: Currently will assume impurity velocity is some fraction of the
            # background velocity (default zero), though this is obviously not true
            # and a better implementation would use the real values wherever they are.
            # This would require having the velocity distribution of the impurities.
            if t13:
                vi = self.read_data_2d_kvhs_t13()
            else:
                # KVHS is scaled by 1/QTIM to get m/s
                vi = self.read_data_2d('KVHS', scaling=1.0 / self.qtim, no_core=no_core)

            vz = vz_mult * vi

            ff = self.crmi * constants.m_p * (vi - vz) / tau_s
            if force.lower() == 'ff':
                return ff

        # TODO: Pressure gradient force calculations
        if force.lower() in ['fpg', 'fnet']:

            print('Pressure gradient force calculation not yet implemented!')
            # Parallel collisional diffusion time
            #tau_par = 1.47E13 * self.crmi * ti * np.sqrt(ti / self.crmb) / \
            #          (ne * np.power(charge, 2) * col_log)
            fpg = 0
            if force.lower() == 'fpg':
                return fpg

        # Electron temperature gradient force calculations
        if force.lower() in ['feg', 'fnet']:
            alpha = 0.71 * np.power(charge, 2)

            # The electron temperature gradient from the code
            # scaling factor as suggested by J. Nichols to get to eV/m
            # KFEGS: electron temperature gradient force component
            kfegs = self.read_data_2d('KFEGS', scaling = constants.e/fact, no_core=no_core)

            # Calculate the force
            feg = alpha * kfegs

            if force.lower() == 'feg':
                return feg

        # Ion temperature gradient force calculations
        if force.lower() in ['fig', 'fnet']:
            mu = self.crmi / (self.crmi + self.crmb) # Stangeby p.315
            beta_i = 3 * (mu + 5 * np.sqrt(2) * np.power(charge, 2) * \
                   (1.1 * np.power(mu, 5/2) - 0.35 * np.power(mu, 3./2)) - 1) / \
                   (2.6 - 2 * mu + 5.4 * np.power(mu, 2))
            print(f"FiG: beta_i = {beta_i:.2f}")

            # scaling factor as suggested by J. Nichols to get to eV/m
            # KFIGS: ion temperature gradient force component
            kfigs = self.read_data_2d('KFIGS', scaling = constants.e / fact, no_core=no_core)

            fig = beta_i * kfigs
            if force.lower() == 'fig':
                return fig

        # Electric field force calculations
        if force.lower() in ['fe', 'fnet']:
            
            E_pol = self.read_data_2d('E_POL', scaling = constants.e / fact, no_core=no_core)
            
            fe = charge * constants.e * E_pol
            if force.lower() == 'fe':
                return fe

        # Net force calculation:
        if force.lower() == 'fnet':
            print('Excluding pressure-gradient force because this is not yet implemented!')
            #return ff + fpg + feg + fig + fe
            return ff + feg + fig + fe

    def plot_2d(self, data, charge=None, scaling=1.0,
                normtype='linear', cmap='plasma', xlim=[0.9, 2.4],
                ylim = [-1.4, 1.4],
                levels=None, cbar_label=None, lut=21,
                smooth_cmap=False, vmin=None, vmax=None,
                no_core=False, vz_mult=0.0, wall_data=None, show_grid=True,
                ax=None):
        """Plot OEDGE results in 2D. This method allows users to plot one of the OEDGE 
        output quantities, or a different arbitrary variable on the same grid.

        Parameters
        ----------
        data: str or array
             Variable name or data array to plot. If a string is given, then attempt to
             fetch the appropriate 2D data. If the data array is directly provided, it is
             assumed that this is in the form returned by `read_data_2d`.
             Some special datanames will perform extra data handling, e.g. KVHSimp.
        charge: int
            The charge state to be plotted, if applicable.
        scaling: float
            Scaling factor to apply to the data, if applicable.
        normtype: str
            One of 'linear', 'log', ... of how to normalize the data on the plot.
        cmap: str
            The colormap to apply to the plot. Uses standard matplotlib names.
        xlim: float
            X range of axes.
        ylim: float
            Y range of axes.
        levels: 1D array or None
            Number of levels for colorbar (needs work).
        cbar_label: str
            Label for the colorbar.
        lut: int
            Number of chunks to break the colormap into.
        smooth_cmap: bool
            Choose whether to break colormap up into chunks or not.
        vmin/vmax:  floats
            Minimum and maximum levels covered by colorbar.
        no_core: bool
            If True, do not include data in the core region in plotting.
        vz_mult: float
            Fraction of the background velocity used for vz (needed only in FF).
        wall_data: (2, npt) array
            Override R,Z coordinates [m] of wall to be plotted.
        show_grid: bool
            If True, show the 2D grid on top of data. 
        ax: matplotlib Axes instance
            If provided, plot on these axes.
        """

        # if data was not directly provided as an array, load the requested variable
        dataname = None
        if isinstance(data, str):
            dataname = copy.deepcopy(data)
            # Read in the data into a form for PolyCollection, accounting for some special options
            
            # Flow velocity with additional velocity specified by T13
            if dataname == 'KVHSimp':
                data = self.read_data_2d_kvhs_t13(no_core=no_core)

            # Special option to plot the ring numbers
            elif dataname == 'Ring':
                data = self.read_data_2d('KTEBS', scaling='Ring', no_core=no_core)
                
            elif dataname == 'Knot':
                data = self.read_data_2d('KTEBS', scaling='Knot', no_core=no_core)

            # Divide the background velocity by the sounds speed to get the Mach number
            elif dataname == 'KVHSimp - Mach':
                te   = self.read_data_2d('KTEBS', no_core=no_core)
                ti   = self.read_data_2d('KTIBS', no_core=no_core)
                kvhs = self.read_data_2d_kvhs_t13()
                mi   = self.crmb * 931.494*10**6 / (3e8)**2  # amu --> eV s2 / m2
                cs   = np.sqrt((te + ti) / mi)
                data = kvhs / cs  # i.e. the Mach number

            elif dataname == 'KVHS - Mach':
                te   = self.read_data_2d('KTEBS', no_core=no_core)
                ti   = self.read_data_2d('KTIBS', no_core=no_core)
                kvhs = self.read_data_2d('KVHS',  no_core=no_core, charge=charge, scaling=scaling)
                mi   = self.crmb * 931.494*10**6 / (3e8)**2  # amu --> eV s2 / m2
                cs   = np.sqrt((te + ti) / mi)
                print(f"CRMB = {self.crmb} amu")
                data = kvhs / cs  # Mach number

            # Special function for plotting the forces on impurities
            elif dataname.lower() in ['ff', 'fig', 'feg', 'fpg', 'fe', 'fnet']:
                if charge is None:
                    raise ValueError('Specify impurity charge for which forces should be computed')
                data = self.calculate_forces(dataname, charge=charge,
                                             no_core=no_core, vz_mult=vz_mult)

            # Everything else in the netCDF file
            else:
                data = self.read_data_2d(dataname, charge, scaling, no_core)

        elif isinstance(data, np.ndarray) and data.shape==self.nc['KNBS']['data'].shape:
            # array was provided in the form of the nc dictionary
            # convert to format for 2D plotting
            data = self.read_data_2d(data)

        else:
            # assume data was already provided in required format
            pass
        
        # Remove any cells that have nan values.
        not_nan_idx = np.where(~np.isnan(data))[0]
        mesh = np.array(self.mesh)[not_nan_idx, :, :]
        data = data[not_nan_idx]

        if not len(data):
            # data matrix is empty. If this is an impurity charge state density,
            # the charge state is probably entirely absent in DIVIMP results
            print("No values to plot")
            return

        if ax is None:
            # Create a good sized figure with correct proportions.
            fig = plt.figure(figsize=(7, 9))
            ax  = fig.add_subplot(111)

        if normtype == 'linear':
            if vmin is None: vmin = data.min()
            if vmax is None: vmax = data.max()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        elif normtype == 'log':
            data[data == 0.0] = sys.float_info.epsilon 
            if vmin is None: vmin = data.min()
            if vmax is None: vmax = data.max()
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        elif normtype == 'symlin':
            if vmin is None: vmin = -np.abs(data).max()
            if vmax is None: vmax = -vmin
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = 'coolwarm'

        elif normtype == 'symlog':
            data[data == 0.0] = sys.float_info.epsilon
            if vmin is None: vmin = -np.abs(data[~np.isnan(data)]).max()
            if vmax is None: vmax = -vmin
            norm = mpl.colors.SymLogNorm(linthresh=0.01 * vmax, vmin=vmin, vmax=vmax, base=10)
            cmap = 'coolwarm'

        # The end of nipy_spectral is grey, which makes it look like there's a
        # hole in the largest data. Fix this by just grabbing a subset of the
        # colormap from 0.0-0.95, leaving out the last portion that's grey
        if cmap == 'nipy_spectral':
            cmap_obj = plt.get_cmap(cmap)
            new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
              'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_obj.name, a=0, b=0.95),
              cmap_obj(np.linspace(0, 0.95, 500)), N=lut)
            cmap = new_cmap

        # Choose whether to discretize the colormap or not first
        scalar_map = mpl.cm.ScalarMappable(
            norm=norm, cmap=cmap if smooth_cmap else mpl.cm.get_cmap(cmap, lut=lut)
        )

        # Create PolyCollection object
        edgecolors="none" if show_grid else "face"
        
        coll = mpl.collections.PolyCollection(mesh, array=data,
                                              cmap=scalar_map.cmap,
                                              norm=scalar_map.norm,
                                              edgecolors=edgecolors)

        # Add the PolyCollection to the Axes object
        ax.add_collection(coll)

        # Plot the wall.
        if wall_data is None:
            # Drop all the (0, 0)'s and append the first point on the end so it
            # doesn't leave a gap in the plot
            keep_idx = np.where(np.logical_and(self.rvesm[0] != 0, self.zvesm[0] != 0))
            rvesm = np.append(self.rvesm[0][keep_idx], self.rvesm[0][keep_idx][0])
            zvesm = np.append(self.zvesm[0][keep_idx], self.zvesm[0][keep_idx][0])
            #ax.plot(self.rvesm[0][keep_idx], self.zvesm[0][keep_idx], color='k', linewidth=1)
        else:
            rvesm = wall_data[0]
            zvesm = wall_data[1]
        ax.plot(rvesm, zvesm, color='k', linewidth=1)

        # Get the separatrix coordinates as a collection of lines and plot
        sep = self.get_sep()
        sc = mpl.collections.LineCollection(sep, color='k')
        ax.add_collection(sc)

        # Use correct number of levels for colorbar, if specified
        cbar = ax.get_figure().colorbar(coll, ax=ax, boundaries=levels, ticks=levels, extend='both')
        
        if (cbar_label is None) and (dataname is not None):
            for key in self.name_maps:
                if self.name_maps[key]['data'] == dataname:
                    cbar_label = self.name_maps[key]['label']
                    if charge is not None:
                        cbar_label += f', Z={charge}' if charge!='all' else '(tot)'
                    cbar_label += f' [{self.name_maps[key]["units"]}]'

        cbar.ax.set_ylabel(cbar_label)

        ax.axis('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        fig.tight_layout()

    def plot_2d_overview(self):
        '''Plot the main background quantities and lowest ionization stages of the
        simulated impurity species.
        '''
        fields = ['ne','Te','Ti','n_H','nimp0','nimp1','nimp2','nimp3']
        fig, axs = plt.subplots(2,4, figsize=(15,8))
      
        for ii,ax in enumerate(axs.flatten()):
            field = fields[ii]
            spl = re.split('(\d+)',field)    
            if len(spl)>1:
                field = spl[0]
                charge = int(spl[1])
            else:
                charge = None
                
            var = self.name_maps[field]['data']
            label = self.name_maps[field]['label']
            units = self.name_maps[field]['units']
            
            self.plot_2d(var, charge=charge, ax=ax, label=label+' '+units)

        plt.tight_layout()

    def load_var_descriptions(self):
        '''Obtain available descriptions of output variables in the NETCDF output file.
        '''
        self.varnames = {}
        for key in self.nc.keys():
            if 'long_name' in self.nc[key] and len(self.nc[key]['long_name']):            
                self.varnames[key] = self.nc[key]['long_name']

    def load_extra_output(self, plot_lp=True):
        """Load additional output data from the .dat file.
        This allows to cross-check the LP data that was given to OEDGE as input and 
        examine results that are not otherwise available in the NETCDF output file.

        Parameters
        ----------
        plot_lp : bool
            If True, plot the input target Langmuir probe data
        """
        # target geometry
        start_line = 'TARGET ELEMENT GEOMETRY:\n\n'
        cut_contents = self.dat_file.split(start_line)[-1].split('\n\n')[0].split('\n')
        header = cut_contents.pop(0).split('  ')
        column_names = [head.strip().lower() for head in header if len(head)]
        data = []
        for line in cut_contents:
            data.append(line.split())
        self.df_geom = pd.DataFrame(data, columns=column_names)
        # change data columns to floats/ints
        for col in ['r','z','psi','length','bth/b','costet','sep dist','mid dist']:
            self.df_geom[col] = pd.to_numeric(self.df_geom[col], downcast='float')
        for col in ['id','ik','ir']:
            self.df_geom[col] = pd.to_numeric(self.df_geom[col], downcast='integer')

        # --------------------------------
        # load LP data given as input
        start_line = 'SUMMARY OF TARGET AND UPSTREAM CONDITIONS FOR THE SELECTED SOL OPTION:\n'
        cut_contents = self.dat_file.split(start_line)[-1].split('\n\n\n')[0].split('\n')
        header = cut_contents.pop(0).split('  ')
        column_names = [head.strip().lower() for head in header if len(head)]
        column_names.insert(1, 'target') # add in/out label to column names
        data = {'inner': [], 'outer': []}
        for line in cut_contents:
            data['inner' if 'INNER' in line.split() else 'outer'].append(line.split())
        self.dfo_lp = pd.DataFrame(data['outer'], columns=column_names)
        self.dfi_lp = pd.DataFrame(data['inner'], columns=column_names)

        # change data columns to floats/ints
        for col in ['te targ','ti targ','nb targ','te mid','ti mid','nb mid']:        
            self.dfo_lp[col] = pd.to_numeric(self.dfo_lp[col], downcast='float')
            self.dfi_lp[col] = pd.to_numeric(self.dfi_lp[col], downcast='float')
        self.dfo_lp['ring'] = pd.to_numeric(self.dfo_lp['ring'], downcast='integer')
        self.dfi_lp['ring'] = pd.to_numeric(self.dfi_lp['ring'], downcast='integer')

        # correct indexing such that we get a continuous profile across PFR and SOL
        idxs = np.concatenate((np.arange(self.irsep)[::-1], np.arange(self.irsep, self.irwall)[::-1]))
        self.dfo_lp = self.dfo_lp.reindex(idxs)
        self.dfi_lp = self.dfi_lp.reindex(idxs)

        # get psin values along targets
        ds_o = np.array(self.df_geom.loc[(self.df_geom['ik']==1)]['sep dist'])
        ds_i = np.array(self.df_geom.loc[(self.df_geom['ik']!=1)]['sep dist'])

        # load conductive power ratios
        start_line = 'SUMMARY OF INTEGRATED CONDUCTIVE POWER RATIOS:\n\n'
        cut_contents = self.dat_file.split(start_line)[-1].split('\n\n\n')[0].split('\n')
        header = cut_contents.pop(0).split()
        column_names = [head.strip().lower() for head in header if len(head)]
        column_names.insert(1, 'target')
        data = {'inner': [], 'outer': []}
        for line in cut_contents:
            data['inner' if 'INNER' in line.split() else 'outer'].append(line.split())
        self.dfo_cond = pd.DataFrame(data['outer'], columns=column_names)
        self.dfi_cond = pd.DataFrame(data['inner'], columns=column_names)
        # change data columns to floats
        for col in ['conde/qetot','condi/qitot','cond/qtot']:
            self.dfo_cond[col] = pd.to_numeric(self.dfo_cond[col], downcast='float')
            self.dfi_cond[col] = pd.to_numeric(self.dfi_cond[col], downcast='float')
        self.dfo_cond['ring'] = pd.to_numeric(self.dfo_cond['ring'], downcast='integer')
        self.dfi_cond['ring'] = pd.to_numeric(self.dfi_cond['ring'], downcast='integer')
        
        if plot_lp:
            fig,axs = plt.subplots(1,2, figsize=(12,5), sharex=True)
            axs[0].plot(ds_o, self.dfo_lp['nb targ'], label='outer target')
            axs[0].plot(ds_i, self.dfi_lp['nb targ'], label='inner target')
            axs[0].set_xlabel('ds [m]')
            axs[0].set_ylabel(r'$n_e$ [m$^{-3}$]')
            axs[0].legend(loc='best').set_draggable(True)

            axs[1].plot(ds_o, self.dfo_lp['te targ'], label='outer target')
            axs[1].plot(ds_i, self.dfi_lp['te targ'], label='inner target')
            axs[1].set_xlabel('ds [m]')
            axs[1].set_ylabel(r'$T_e$ [eV]')
            axs[1].legend(loc='best').set_draggable(True)

        
    def _load_kvhs_adj_2d(self):
        '''Load the 2D array of KVHS with the additional drifts added on.
        '''
        # Get the 2D data from the netCDF file.
        scaling  = 1.0 / self.qtim
        kvhs     = self.nc['KVHS']['data'] * scaling

        # Array to hold data with T13 data added on (will be same as kvhs if T13 was off).
        kvhs_adj = kvhs

        try:
            pol_opt = float(self.dat_file.split('POL DRIFT OPT')[1].split(':')[0])
            if pol_opt == 0.0:
                print('Poloidal drift option T13 was OFF.')

            elif pol_opt in [1.0, 2.0, 3.0]:
                print('Poloidal drift option T13 was ON.')

                # Get the relevant table for the extra drifts out of the .dat file.
                add_data = self.dat_file.split('TABLE OF DRIFT REGION BY RING - RINGS ' + \
                                                'WITHOUT FLOW ARE NOT LISTED\n')[1]. \
                                                split('DRIFT')[0].split('\n')

                # Split the data between the spaces, put into DataFrame.
                add_data = [line.split() for line in add_data]
                add_df = pd.DataFrame(add_data[1:-1], columns=['IR', 'Vdrift (m/s)',
                                      'S_START (m)', 'S_END (m)'], dtype=np.float64). \
                                      set_index('IR')

                # Loop through the KVHS data one cell at a time, and if
                # the cell has extra Mach flow, add it.
                for ir in range(self.nrs):
                    for ik in range(self.nks[ir]):
                        if self.area[ir, ik] != 0.0:

                            # If this ring has additional drifts to be added.
                            if ir+1 in add_df.index:  # ir is zero-indexed

                                # Then add the drift along the appropriate s (or knot) range.
                                if self.kss[ir][ik] > add_df['S_START (m)'].loc[ir+1] and \
                                   self.kss[ir][ik] < add_df['S_END (m)'].loc[ir+1]:

                                   kvhs_adj[ir][ik] = kvhs[ir][ik] + add_df['Vdrift (m/s)'].loc[ir+1]
            return kvhs_adj

        except AttributeError:
            print("Error: .dat file has not been loaded.")

        except IndexError:
            # Happens if DIVIMP not run, and since T13 is a DIVIMP only option,
            # it's irrelevant when just creating background.
            return kvhs_adj
            
    def along_ring(self, ring, dataname,
                   ylabel=None, charge=None, vz_mult=0.0, plot=True):
        """Plot data along a specified ring.

        Parameters
        ----------
        ring: int
            The ring number to plot data for.
        dataname: str
            The NetCDF variable you want the dat along the ring for. 
            Special options include 'Mach' or 'Velocity' that can add 
            on the additional drift option T13 if it was on.
        ylabel: str
            Label for the Y-axis.
        charge: str
            Charge, if needed.
        vz_mult: float
            The multiplier to be used in FF calculations 
            (see `~aurora.oedge.oedge_case.calculate forces`).
        plot: bool
            If True, plot the data along the chosen ring.

        Returns
        -------
        s : 1D array
            Value of the s coordinate along field lines.
        data : 1D array
            Extracted data along the s coordinate.
        """
        # Get the parallel to B coordinate (ignore first data point)
        x = self.nc['KSB']['data'][ring-1, 1:]
        
        # Some translations
        if dataname in ['KVHS - Mach', 'KVHSimp - Mach']:
            dataname = 'Mach'
            
        if dataname in ['KVHS', 'KVHSimp']:
            dataname = 'Velocity'

        if dataname in ['Mach', 'Velocity']:
            # check if T13 option was on to get Mach number or speed
            kvhs_adj = self._load_kvhs_adj_2d()

            # Finally put it into the y value of the ring we want
            if dataname == 'Mach':

                # Need to calculate the sound speed to back out the Mach number
                te = self.nc['KTEBS']['data'][ring-1]
                ti = self.nc['KTIBS']['data'][ring-1]
                mb = self.crmb * 931.49 * 10**6 / ((3*10**8)**2)
                cs = np.sqrt((te + ti) / mb)
                y  = kvhs_adj[ring-1] / cs

            elif dataname == 'Velocity':
                y  = kvhs_adj[ring-1]  # m/s

        elif dataname =='DDLIMS':
            # impurity densities -- zero index to be ignored (primary neutrals)
            scaling = self.absfac
            if charge in ['all', None]:
                y = self.nc[dataname][1:].sum(axis=0)[ring-1] * scaling
            else:
                y = self.nc[dataname][1+charge][ring-1] * scaling

        elif dataname.lower() in ['ff', 'fig', 'feg', 'fpg', 'fe', 'fnet', 'ff']:

            # Some constants and required factors.
            col_log = 15
            fact = np.power(self.qtim, 2) * (constants.e / constants.m_p) / self.crmi

            if dataname.lower() in ['fig', 'fnet']:

                # Need charge to calculate FiG.
                if charge is None:
                    print("Error: Must supply a charge state to calculate FiG")
                    return None

                #mu = self.crmi / (self.crmi + self.crmb)
                mu = self.crmi*self.crmb / (self.crmi + self.crmb) # reduced mass
                beta = 3 * (mu + 5 * np.sqrt(2) * np.power(charge, 2) * \
                       (1.1 * np.power(mu, 5/2)- 0.35 * np.power(mu, 3/2)) - 1) / \
                       (2.6 - 2 * mu + 5.4 * np.power(mu, 2))
                print("FiG: Beta = {:.2f}".format(beta))

                # KFIGS: Ion temperature gradient force component
                kfigs = self.nc['KFIGS']['data'][ring-1] * constants.e / fact

                # Calculate the force.
                fig = beta * kfigs
                y = np.array(fig, dtype=np.float64)

            if dataname.lower() in ['ff', 'fnet']:

                # Need charge to calculate FF.
                if charge is None:
                    print("Error: Must supply a charge state to calculate FF")
                    return None

                ti = self.nc['KTIBS']['data'][ring-1]
                ne = self.nc['KNBS']['data'][ring-1]

                # Slowing down time
                tau_s = 1.47E13 * self.crmi * ti * np.sqrt(ti / self.crmb) / \
                        ((1 + self.crmb / self.crmi) * ne * np.power(charge, 2) * col_log)

                # TODO: Currently will assume impurity velocity is some fraction of the
                # background velocity (default zero), though this is obviously not true
                # and a better implementation would use the real values wherever they are.

                kvhs_adj = _load_kvhs_adj_2d()
                vi = kvhs_adj['data'][ring-1]
                vz = vz_mult * vi

                # Calculate the force.
                ff = self.crmi * constants.m_p * (vi - vz) / tau_s
                y = np.array(ff, dtype=np.float64)

            if dataname.lower() in ['fe', 'fnet']:

                e_pol = self.nc['E_POL']['data'][ring-1] * constants.e / fact
                fe = charge * constants.e * e_pol
                y = np.array(fe, dtype=np.float64)

            if dataname.lower() in ['feg', 'fnet']:

                alpha = 0.71 * np.power(charge, 2)

                # The electron temperature gradient from the code - convert to eV/m
                kfegs = self.nc['KFEGS']['data'][ring-1] * constants.e / fact

                # Calculate the force.
                feg = alpha * kfegs
                y = np.array(feg, dtype=np.float64)

            if dataname.lower() in ['fpg', 'fnet']:

                # Parallel collisional diffusion time
                tau_par = 1.47E13 * self.crmi * ti * np.sqrt(ti / self.crmb) / \
                          (ne * np.power(charge, 2) * col_log)

                raise ValueError('FPG not yet implemented!')
                #y = fpg

            if dataname.lower() == 'fnet':
                y = fig + ff + feg + fe + fpg

        else:
            # Get the data for this ring
            if charge is None:
                y = np.array(self.nc[dataname]['data'][ring-1], dtype=np.float64)
            else:
                y = np.array(self.nc[dataname]['data'][charge-1][ring-1], dtype=np.float64)

        # Remove any (0, 0) data points
        drop_idx = np.array([], dtype=np.int)
        for i in range(0, len(x)):
             if x[i] == 0.0 and y[i] == 0.0:
                 drop_idx = np.append(drop_idx, i)
        x = np.delete(x, drop_idx)
        y = np.delete(y, drop_idx)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, y, 'k-')
            ax.set_xlabel(r's [m]')
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='both')
            fig.tight_layout()

        return x, y

    def input_for_midpoint_shift(self, knot, plot=True):
        """Print out data that you can input into the OEDGE input file for the shifted 
        midpoint option G55.

        One may run this method iteratively until a satisfactory knot is found.
        
        Parameters
        ----------
        knot : int
            Knot you want the midpoint shifted to for each ring in just the main SOL.
        plot : bool
            If True, plot 2D contour showing the selected knot
        """
        if plot:
            # First plot the knot to make sure it's the one you want the midpoint to be at
            self.plot_2d('Knot', vmin=knot-1, vmax=knot+1, lut=3)

        # Load the Snorm data in 2D format
        snorm = self.nc['KSB']['data'] / self.ksmaxs[:, None]
        snorm = np.abs(2*snorm - 1.0)   # check

        if float(self.nc["ZXP"]['data']) > 0:
            soffset_mult = -1.0  # USN plasma
        else:
            soffset_mult = 1.0  # LSN plasma

        output = []
        # add the offset for each ring one at a time
        for ring in range(self.irsep, self.irwall+1):
            soffset = soffset_mult * snorm[ring-1][knot-1] / 2.0  # Ring and knot are 1-indexed here
            #print("{:8}{:8}{:8}{:10.6f}".format(1, ring, ring, soffset))
            output.append([1,ring,ring,soffset])
            
        return output

    def find_knot_on_ring(self, ring, r, z):
        """Find the closest knot on a ring to an (R, Z).

        Parameters
        ----------
        ring: int
            Ring to find knot on.
        r : float
            R location of the point on the grid.
        z : float
            Z location of the point on the grid.

        Returns
        -------
        closest_knot: int
            The knot on this ring closest to r, z.
        """
        dist = np.sqrt((r - self.rs[ring])**2 + (z - self.zs[ring])**2)
        closest_knot = np.where(dist == dist.min())
        return closest_knot[0][0]

    def find_ring_knot(self, r, z):
        """Find the ring and the knot on that ring of the cell
        that are closest to the input (r, z).

        Parameters
        ----------
        r : float
            R location of the point on the grid.
        z : float
            Z location of the point on the grid.

        Returns
        -------
        ring, knot : int, int
            The ring and knot of which cell this point is in.
        """
        dist = np.sqrt((r - self.rs)**2 + (z - self.zs)**2)
        closest_cell = np.where(dist == dist.min())

        return (closest_cell[0][0], closest_cell[1][0])

    def find_ring_from_psin(self, psin):
        """
        Helper function to find the ring with the closest average psin value to
        the input psin in.

        TODO: make sure that identified psin values are from the core and not from
        the PFR or other rings near wall structures.

        Parameters
        ----------
        psin : float
            Psin value of which to find the closest ring for.

        Returns
        -------
        close_ring : int
            The closest ring index to this psin.
        """
        # See if we've already calculated psin_dict.
        if not hasattr(self, 'psin_dict'):
            self.psin_dict = {}
            psifl = np.array(self.nc.variables['PSIFL']['data'])
            for ring in range(0, psifl.shape[0]):
                if np.count_nonzero(psifl[ring]) == 0:
                    continue
                else:

                    # Store the average psin value into our dictionary.
                    psin_avg = psifl[ring][psifl[ring] != 0].mean()
                    self.psin_dict[ring] = psin_avg

        # Elegant one-liner to find closest ring to psin.
        close_ring, _ = min(self.psin_dict.items(), key=lambda item: abs(item[1] - psin))
        return close_ring

    def mock_probe(self, r_start, r_end, z_start, z_end, data='Te', num_locs=100,
                   plot=None, show_plot=True):
        """Synthetic probe measurement at specified locations.

        Parameters
        ----------
        r_start: float
            R coordinate of the measurement starting point.
        r_end: float
            R coordinate of the measurement ending point.
        z_start: float
            Z coordinate of the measurement starting point.
        z_end: float
            Z coordinate of the measurement ending point.
        data: str
            One of 'Te', 'ne', 'Mach', 'Velocity', 'L OTF', or 'L ITF'.
        num_locs: int
            Spatial discretization between input R and Z points.
        plot: str or None
            Either None, 'R' or 'Z' (or 'r' or 'z'), or 'psin'. If the probe is at a
                     constant R, then use 'R', likewise for 'Z'.
        show_plot: bool
            Show the plot or not (i.e. if you just want the data).

        Output
        x, y : The data used in the plot that simulates, for example, a plunging
                Langmuir probe or something.
        """
        # Create rs and zs to get measurements at.
        rs = np.linspace(r_start, r_end, num_locs)
        zs = np.linspace(z_start, z_end, num_locs)

        # DataFrame for all the output.
        output_df = pd.DataFrame(columns=['(R, Z)', 'Psin', data], index=np.arange(0, num_locs))

        # If we want the Mach number (or speed), we need to do a little data
        # preprocessing first to see if the additional T13 drift option was on.
        if data in ['Mach', 'Velocity']:

            # Get the 2D data from the netCDF file.
            scaling  = 1.0 / self.qtim
            kvhs     = self.nc['KVHS']['data'] * scaling

            # Array to hold data with T13 data added on (will be same as
            # kvhs if T13 was off).
            kvhs_adj = kvhs

            # See if T13 was on and additional values need to be added.
            try:
                pol_opt = float(self.dat_file.split('POL DRIFT OPT')[1].split(':')[0])
                if pol_opt == 0.0:
                    print('Poloidal drift option T13 was OFF.')

                elif pol_opt == 1.0:
                    print('Poloidal drift option T13 was ON.')

                    # Get the relevant table for the extra drifts out of the .dat file.
                    add_data = self.dat_file.split('TABLE OF DRIFT REGION BY RING - RINGS ' + \
                                                    'WITHOUT FLOW ARE NOT LISTED\n')[1]. \
                                                    split('DRIFT')[0].split('\n')

                    # Split the data between the spaces, put into DataFrame.
                    add_data = [line.split() for line in add_data]
                    add_df = pd.DataFrame(add_data[1:-1], columns=['IR', 'Vdrift (m/s)',
                                          'S_START (m)', 'S_END (m)'], dtype=float). \
                                          set_index('IR')

                    # Loop through the KVHS data one cell at a time, and if
                    # the cell has extra Mach flow, add it.
                    for ir in range(self.nrs):
                        for ik in range(self.nks[ir]):
                            if self.area[ir, ik] != 0.0:

                                # If this ring has additional drifts to be added...
                                if ir in add_df.index:

                                    # ...then add the drift along the appropriate s (or knot) range.
                                    if self.kss[ir][ik] > add_df['S_START (m)'].loc[ir] and \
                                       self.kss[ir][ik] < add_df['S_END (m)'].loc[ir]:

                                       kvhs_adj[ir][ik] = kvhs[ir][ik] + add_df['Vdrift (m/s)'].loc[ir]

            except AttributeError:
                print("Error: .dat file has not been loaded.")
            except IndexError:
                print("Warning: Can't add on T13 data if DIVIMP is not run.")

            # Fill in the psin values for the dataframe
            for i in range(0, len(rs)):
                ring, knot = self.find_ring_knot(rs[i], zs[i])
                psin = self.nc['PSIFL']['data'][ring][knot]
                output_df.iloc[i]['Psin'] = psin

        for i in range(0, num_locs):

            # Get the cell that has the data at this R, Z
            ring, knot = self.find_ring_knot(rs[i], zs[i])

            if data == 'Te':
                probe = self.nc['KTEBS']['data'][ring][knot]
                ylabel = 'Te (eV)'

            elif data == 'ne':
                probe = self.nc['KNBS']['data'][ring][knot]
                ylabel = 'ne (m-3)'

            elif data == 'Mach':
                # Need to calculate the sound speed to back out the Mach number
                te = self.nc['KTEBS']['data'][ring][knot]
                ti = self.nc['KTIBS']['data'][ring][knot]
                mb = self.crmb * 931.49 * 10**6 / ((3*10**8)**2)
                cs = np.sqrt((te + ti) / mb)
                probe = kvhs_adj[ring][knot] / cs
                ylabel = 'Mach'

            elif data == 'Velocity':
                probe = kvhs_adj[ring][knot]
                ylabel = 'Velocity (m/s)'

            # Plot of the connection length to the inner target.
            elif data == 'L OTF':
                smax = self.nc['KSMAXS']['data'][ring]
                s    = self.nc['KSS']['data'][ring][knot]
                probe = smax - s
                ylabel = 'L ITF (m)'

            elif data == 'L ITF':
                s    = self.nc['KSS']['data'][ring][knot]
                probe = s
                ylabel = 'L OTF (m)'

            output_df['(R, Z)'][i] = (rs[i], zs[i])
            output_df[data][i]   = probe

        if plot is not None:

            # Get correct X and Y arrays for plotting.
            if plot.lower() in ['r', 'rminrsep']:
                x = [output_df['(R, Z)'][i][0] for i in range(0, len(output_df.index))]
                xlabel = 'R (m)'
                if plot.lower() == 'rminrsep':
                    pass
            elif plot.lower() == 'z':
                x = [output_df['(R, Z)'][i][1] for i in range(0, len(output_df.index))]
                xlabel = 'Z (m)'
            elif plot.lower() == 'psin':
                x = output_df['Psin'].values
                xlabel = 'Psin'
            y = output_df[data].values

            if show_plot:
                fig = plt.figure()
                ax  = fig.add_subplot(111)
                ax.plot(x, y, lw=3, color='k')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.tick_params(axis='both')
                fig.tight_layout()

        return x, y

    
    def get_3d_path(self, pnt1, pnt2, npt=501, plot=False, ax=None):
        """Given 2 points in 3D Cartesian coordinates, returns discretized 
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
            a new figure and also draw the grid polygons.

        Returns
        -------
        pathR : array (npt,)
            R [m] points along the path.
        pathZ : array (npt,)
            Z [m] points along the path.
        pathL : array (npt,)
            Length [m] discretization along the path.
        """
        xyz = np.outer(pnt2 - pnt1, np.linspace(0, 1, int(npt))) + pnt1[:, None]

        # mapping x, y, z to poloidal plane
        pathR, pathZ = np.hypot(xyz[0], xyz[1]), xyz[2]

        pathL = np.linalg.norm(xyz - pnt1[:, None], axis=0)

        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, figsize=(9, 11))

                # get polygons describing OEDGE grid
                p = mpl.collections.PolyCollection(
                    self.mesh, edgecolors='k', fc='w', linewidth=0.1)
                
                ax.add_collection(p)
                ax.set_xlabel("R [m]")
                ax.set_ylabel("Z [m]")
                ax.axis("scaled")

            # now overplot
            ax.plot(pathR, pathZ, c="b")

        return pathR, pathZ, pathL

    def eval_LOS(self, pnt1, pnt2, vals,
                 npt=501, method="linear", plot=False, ax=None, label=None):
        """Evaluate the OEDGE output `field` along the line-of-sight (LOS)
        given by the segment going from point `pnt1` to point `pnt2` in 3D
        geometry.

        Parameters
        ----------
        pnt1 : array (3,)
            Cartesian coordinates x,y,z of first extremum of LOS.
        pnt2 : array (3,)
            Cartesian coordinates x,y,z of second extremum of LOS.
        vals : array (NY, NX)
            Data array for a variable of interest. This is expected to be given
            on the 2D grid in the nc file.
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
            Values of requested OEDGE output along the LOS
        """
        # get R,Z and length discretization along LOS
        pathR, pathZ, pathL = self.get_3d_path(pnt1, pnt2, npt=npt, plot=False)

        # interpolate from cell centers onto LOS -- use only filled grid
        mask = (self.rs!=0)\
               & (self.rs>np.min(pathR)) & (self.rs<np.max(pathR))\
               & (self.zs>np.min(pathZ)) & (self.zs<np.max(pathZ))

        if np.sum(mask)==0:
            print('No data along given line of sight!')
            return np.nan

        try:
            vals_LOS = griddata(
                (self.rs[mask], self.zs[mask]),
                vals[mask],
                (pathR, pathZ),
                method=str(method),
            )
        except:
            # no data within selected LOS
            vals_LOS = np.zeros_like(pathR)
            
        if plot:
            # plot values interpolated along LOS
            if ax is None:
                fig, ax1 = plt.subplots()
                ax1.set_xlabel("l [m]")
            else:
                ax1 = ax
            ax1.plot(pathL, vals_LOS, label=label)
            ax1.legend(loc="best").set_draggable(True)
            plt.tight_layout()

        return vals_LOS

    def calc_rad(self,
                 bckg=True,
                 adas_files_sub={},
                 sxr_flag=False,
                 thermal_cx_rad_flag=False,
                 ):
        """Calculate radiation from ADAS ADF11 files using Aurora, either for the background species
        (assumed of hydrogen isotopes) or the simulated impurity.

        Parameters
        ----------
        bckg : bool
            If True, compute radiation for the background species, assumed to be an hydrogen 
            isotope. If False, compute radiation for the simulated impurity.
        adas_files_sub : dict
            Dictionary containing ADAS file names for radiation calculations, possibly including keys
            "plt","prb","prc","pls","prs","pbs","brs"
            Any file names that are needed and not provided will be searched in the 
            :py:meth:`~aurora.adas_files.adas_files_dict` dictionary.         
        sxr_flag : bool, optional
            If True, soft x-ray radiation is computed (for the given 'pls','prs' ADAS files)  
        thermal_cx_rad_flag : bool, optional
            If True, thermal charge exchange radiation is computed. 
            NB: this is redundant if the background (H) radiation is calculated.
        
        Returns
        -------
        dict :
            Dictionary containing output of :py:meth:`~aurora.radiation.compute_rad`.
            Please refer to this method for info.
        """
        # only do calculation for cells with non-zero area
        #mask = self.area != 0
        
        if bckg:
            # array of H and proton densities; cs must be middle index
            nz = np.stack((self.nc['PINATO']['data'], self.nc['KNBS']['data'])).transpose(1,0,2)
        else:
            # find impurity species atomic symbol
            from omfit_classes import utils_math
            key = list(utils_math.atomic_element(Z=self.cion).keys())[0]
            imp = utils_math.atomic_element(Z=self.cion)[key]['symbol']
            nz = self.nc['DDLIMS']['data'][1:].transpose(1,0,2)  # first index to be ignored
            
        return radiation.compute_rad('H' if bckg else imp,
                                  nz*1e-6, # m^-3-->cm^-3
                                  self.nc['KNBS']['data']*1e-6, # m^-3-->cm^-3
                                  self.nc['KTEBS']['data'],  # eV
                                  n0=self.nc['PINATO']['data']*1e-6, # m^-3-->cm^-3
                                  Ti=self.nc['KTIBS']['data'],  # eV
                                  adas_files_sub=adas_files_sub,
                                  prad_flag=True,
                                  sxr_flag=sxr_flag,
                                  thermal_cx_rad_flag=thermal_cx_rad_flag
        )

    def get_outer_midplane_prof(self, varname='ne', plot=False):
        '''Extract outer midplane profiles via linear interpolation of 2D output profiles.

        Parameters
        ----------
        varname : str
             Name of variable. This can be either in "simple nomenclature" (e.g. "ne"),
             or in OEDGE nomenclature (e.g. "KNBS").
        plot : bool
             If True, midplane profiles are plotted.
        '''
        var = self.name_maps[varname]['data'] if varname in self.name_maps else varname
        var_label = fr'{self.name_maps[varname]["label"]} '+\
                    '[{self.name_maps[varname]["units"]}]'
        vals = self.nc[var]['data']

        # magnetic axis radius
        R0 = self.nc['R0']['data']
        rsep = self.rvertp[self.korpg[self.irsep-1,:self.nks[self.irsep-1]]][:,0]
        zsep = self.zvertp[self.korpg[self.irsep-1,:self.nks[self.irsep-1]]][:,0]
        Rsep = interp1d(zsep[rsep>R0], rsep[rsep>R0])(0)

        # create vector of outer midplane R
        Rvec = np.linspace(Rsep-0.05, Rsep+0.1, 201)

        # interpolate output variable onto this R grid
        mask = self.rs != 0
        vals_mid = griddata(
            (self.rs[mask], self.zs[mask]),
            vals[mask],
            (Rvec, np.zeros_like(Rvec)),
            method='linear',
        )

        # get psiN coord along same radial cut
        psiN = griddata(
            (self.rs[mask], self.zs[mask]),
            self.nc['PSIFL']['data'][mask],
            (Rvec, np.zeros_like(Rvec)),
            method='linear',
        )
        
        if plot:
            fig,ax = plt.subplots()
            ax.plot(Rvec-Rsep, vals_mid)
            ax.set_xlabel('R [m]')
            ax.set_ylabel(var_label)

        profs = {'ds': Rvec-Rsep, 'psiN': psiN, 'R': Rvec, 'data': vals_mid}

        return profs

    def get_target_profs(self, varname='ne', plot=False):
        '''Extract target profiles, on both the inner and outer divertor.

        Parameters
        ----------
        varname : str
             Name of variable. This can be either in "simple nomenclature" (e.g. "ne"),
             or in OEDGE nomenclature (e.g. "KNBS").
        plot : bool
             If True, midplane profiles are plotted.
        '''
        # select requested data
        var = self.name_maps[varname]['targ'] if varname in self.name_maps else varname
        var_label = fr'{self.name_maps[varname]["label"]} [{self.name_maps[varname]["units"]}]'

        # NOTE: this explains use of ikds and idds
        #for ir in np.arange(self.irsep,self.nrs):
        #    print(ir,self.nks[ir], ikds[idds[0,ir]-1])

        _res = {}
        for field in ['data','R','Z','ds','psiN']:
            _res[field] = np.zeros((2,self.nrs-self.irsep+1))

        # PSIn values of target elements
        for ii,ir in enumerate(np.arange(self.irsep-1, self.nrs)):
            for jj,ik in enumerate([0, self.nks[ir]]):
                tid =  self.nc['IDDS']['data'][jj,ir]-1  # python indexing
                _res['psiN'][jj,ii] = self.nc['PSIFL']['data'][ir,ik]
                _res['R'][jj,ii] = self.nc['RP']['data'][tid]
                _res['Z'][jj,ii] = self.nc['ZP']['data'][tid]
                _res['ds'][jj,ii] = self.nc['SEPDIS']['data'][tid]
                _res['data'][jj,ii] = self.nc[var]['data'][tid]

        # mask elements with no data (usually, irwall and irwall-1)
        res = {}
        for field in ['data','R','Z','ds','psiN']:
            res[field] = np.zeros((2, np.sum(_res['Z'][1]!=0)))
            res[field][0] = _res[field][0][_res['Z'][1]!=0]
            res[field][1] = _res[field][1][_res['Z'][1]!=0]
            
        # re-order by value of psiN
        idx = np.argsort(res['psiN'], axis=1)
        for field in ['data','R','Z','ds','psiN']:
            res[field][0] = res[field][0,idx[0]]
            res[field][1] = res[field][1,idx[1]]

        # SEPDIS is only given as an absolute value. Add appropriate sign:
        rsep = self.rvertp[self.korpg[self.irsep-1,:self.nks[self.irsep-1]]][:,0]
        zsep = self.zvertp[self.korpg[self.irsep-1,:self.nks[self.irsep-1]]][:,0]
        # index 0 is outer, 1 is inner.
        # rsep,zsep have 0th index on HFS, last index on LFS
        res['ds'][0] = np.sign(res['Z'][0]-zsep[-1])*\
                       np.hypot(res['R'][0]-rsep[-1], res['Z'][0]-zsep[-1])
        res['ds'][1] = np.sign(res['Z'][1]-zsep[0])*\
                       np.hypot(res['R'][1]-rsep[0], res['Z'][1]-zsep[0])
        
        if plot:

            # plot as a function of psin
            fig,ax = plt.subplots()
            ax.plot(res['psiN'][0], res['data'][0], label='outer')
            ax.plot(res['psiN'][1], res['data'][1], label='inner')
            ax.legend(loc='best').set_draggable(True)
            ax.set_xlabel(r'$\psi_N$')
            ax.set_ylabel(var_label)
            
            # plot target profiles as a function of ds
            fig, ax = plt.subplots()
            ax.plot(res['ds'][0], res['data'][0], '.', label='outer')
            ax.plot(res['ds'][1], res['data'][1], '.', label='inner')
            ax.legend(loc='best').set_draggable(True)
            ax.set_xlabel('ds [m]')
            ax.set_ylabel(var_label)

            # plot as a function of Z
            fig, ax = plt.subplots()
            ax.plot(res['Z'][0], res['data'][0], '.', label='outer')
            ax.plot(res['Z'][1], res['data'][1], '.', label='inner')
            ax.legend(loc='best').set_draggable(True)
            ax.set_xlabel('Z [m]')
            ax.set_ylabel(var_label)

        profs = {
            'in': {'R': res['R'][1], 'Z': res['Z'][1], 'ds': res['ds'][1],
                   'psiN': res['psiN'][1], 'data': res['data'][1]},
            'out': {'R': res['R'][0], 'Z': res['Z'][0], 'ds': res['ds'][0],
                    'psiN': res['psiN'][0], 'data': res['data'][0]}
        }
                 
        return profs
                
    def plot_summary(self, varnames=['ne','Te','Ti','Vb'], coord='ds', ls = '-'):
        '''Plot a summary of midplane and target profiles.
        
        Parameters
        ----------
        varnames : list
            Names of variables, in "simple nomenclature", of which summary
            profiles should be plotted. Default is to plot 'ne','Te','Ti' 
            and 'Vb' (parallel velocity).
        coord : str
            Normalized coordinate to plot against. One of ['ds','psiN']
        ls : str
            Style of plotting to be used for all figures. This can include
            also a color in the string. 
        '''
        if coord not in ['ds','psiN']:
            raise ValueError('Only ds and psiN accepted as coordinates for summary plot!')
        
        # collect all profiles
        profs = {'tar':{}, 'mid':{}}
        for varname in varnames:
            profs['tar'][varname] = self.get_target_profs(varname)
            profs['mid'][varname] = self.get_outer_midplane_prof(varname)

        # now plot all profiles
        fig, axs = plt.subplots(len(varnames), 3,
                                figsize=(3*len(varnames), 8),
                                sharex=True, sharey='row')
        axs[0, 0].set_title("Inner target")
        for i,varname in enumerate(varnames):
            axs[i, 0].plot(profs['tar'][varname]['in'][coord],
                           profs['tar'][varname]['in']['data'], ls)

        axs[0, 1].set_title("Outer midplane")
        for i, varname in enumerate(varnames):
            axs[i, 1].plot(profs['mid'][varname][coord], profs['mid'][varname]['data'], ls)

        axs[0, 2].set_title("Outer target")
        for i, varname in enumerate(varnames):
            axs[i, 2].plot(profs['tar'][varname]['out'][coord],
                           profs['tar'][varname]['out']['data'], ls)

        # loop over variables to set labels and grids
        for i, varname in enumerate(varnames):
            var_label = fr'{self.name_maps[varname]["label"]} '+\
                        f'[{self.name_maps[varname]["units"]}]'
            axs[i, 0].set_ylabel(var_label)
            for jj in [0, 1, 2]:
                axs[i, jj].grid(True, ls="--")
        for jj in [0, 1, 2]: axs[-1, jj].set_xlabel("ds")
        
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        plt.tight_layout()



def interpret(x):

    # import here to avoid issues during regression tests
    from omfit_classes.omfit_namelist import namelist
        
    x = x.split('\'')
    for k in range(1, len(x), 1)[::2]:
        x[k] = '\'' + x[k] + '\''
    x = [_f for _f in x if _f]

    xx = []
    for k in x:
        if k.startswith('\''):
            xx.append(k)
        else:
            xx.extend(list(map(namelist.interpreter, [_f for _f in re.split(' |\t', k) if _f])))

    data = []
    inline_comment = ''

    for k in range(len(xx)):
        if not len(inline_comment) and isinstance(xx[k],(int,float)):
            data.append(xx[k])
        elif not len(inline_comment) and xx[k][0] == '\'':
            data.append(xx[k][1:-1])
        else:
            inline_comment += ' ' + str(xx[k])
    inline_comment = inline_comment.strip(' \'')

    return data, inline_comment


