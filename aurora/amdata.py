'''Tools to download, read and process data from the AMJUEL and HYDHEL databases, obtained directly from the eirene.de website.
All reactions are read from the tex files of available documentation, parsed to form Python dictionaries that are stored in a 
json file.

Contributors: T. Lunt, F. Sciortino (MPI-IPP)
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import sys, os
import requests
import json
import copy

local_path = os.path.dirname(os.path.realpath(__file__))

vnfdEnergy = [
    ["emin", "Emin", 1.0, None],
    ["emax", "Emax", 1.0, None],
    ["eth", "Eth", 1.0, None],
]
vnfdTn = [
    ["t1min=", "Tmin", 1.0, 0.1],
    ["t1max=", "Tmax", 1.0, 1e4],
    ["n2min=", "nmin", 1e6, 1e10],
    ["n2max=", "nmax", 1e6, 1e25],
]

class reactions_database:
    '''Class to read and evaluate rates from the AMJUEL and HYDHEL databases.
    '''
    def __init__(self, fn=None):
        self.db = None
        self.c = None
        self.latex = ""
        self.verbose = True
        self.jreaction = None
        if not (fn is None):
            self.open_database(fn)
        self.minT = None
        self.maxT = None

        # load database (download and parse automatically if needed)
        self.open_database()
        
    def open_database(self):

        json_filepath = local_path + os.sep + 'amjuel_hydhel_database.json'

        # if not available, create json file containing all reactions in python dictionary form
        if not os.path.exists(json_filepath):
            
            # populate dictionary with all AMJUEL+HYDHEL reactions
            database = {}
            database.update(parse_amjuel())
            database.update(parse_hydhel())

            # store locally as json file
            json.dump(
                database, open(json_filepath, 'w'), indent=4, sort_keys=True
            )

        self.db = json.load(open(json_filepath))

    def select_reaction(self, rhn=None, report=None, header=None, name=None):
        if self.db is None:
            raise Exception("You need to call open_database first.")
        if rhn is None:
            rhn = "%s,%i,%s" % (report, int(header), name)

        rhn = rhn.replace(
            ".", "_"
        )  # the symbol . is used by json/fson to address a child element

        self.jreaction = self.db[rhn]

        self.c = np.array(self.jreaction["coefficients"])

        for n in [
            "Tmin",
            "Tmax",
            "nmin",
            "nmax",
            "Emin",
            "Emax",
            "factor_n",
            "factor",
            "latex",
            "symbol",
            "unit",
            "parameters",
            "name",
            "header",
            "report",
        ]:
            setattr(self, n, self.jreaction[n] if n in self.jreaction.keys() else None)

        return True

    def reaction(self, n=None, T=None, E=None):
        if self.c is None:
            raise Exception("You need to call select reaction first.")
        params = self.parameters  # shorter notation
        if ("n" in params) and (n is None):
            raise Exception("You need to specify a density.")
        if ("T" in params) and (T is None):
            raise Exception("You need to specify a Temperature.")
        if ("E" in params) and (E is None):
            raise Exception("You need to specify an energy.")

        # validity checks
        if "E" in params:
            E = np.array(E) * 1.0
            if not (self.Emin is None):
                E[E < self.Emin] = np.nan
            if not (self.Emax is None):
                E[E > self.Emax] = np.nan

        if "T" in params:
            T = np.array(T) * 1.0
            if not (self.Tmin is None):
                T[T < self.Tmin] = np.nan
            if not (self.Tmax is None):
                T[T > self.Tmax] = np.nan

        if "n" in params:
            n = np.array(n) * 1.0
            if not (self.nmin is None):
                n[n < self.nmin] = np.nan
            if not (self.nmax is None):
                n[n > self.nmax] = np.nan

        if params.count(",") == 0:  # single-polynomial fits
            if params == "E":
                logx = np.log(E)
            elif params == "T":
                logx = np.log(T)
            else:
                raise Exception("Parameter dependence not implemented.")

            f = logx * 0.0  # create an empty array of the same size than x
            for i in range(9):
                f += self.c[i] * logx ** i
            return np.exp(f) * self.factor

        elif params.count(",") == 1:  # double-polynomial fits
            if params == "E,T":
                logx = np.log(E)
                logy = np.log(T)
            if params == "n,T":
                logx = np.log(n * self.factor_n)
                logy = np.log(T)

            f = logx * 0.0  # create an empty array of the same size than x
            for ix in range(9):
                fy = logy * 0.0
                for iy in range(9):
                    fy += self.c[iy, ix] * logy ** iy
                f += fy * logx ** ix

            return np.exp(f) * self.factor

        else:
            raise Exception(
                "Only one reaction with single- or double polynomial fits are implemented so far."
            )




class h_am_pecs:
    '''Analysis of atomic and molecular emission for H-isotope spectral lines (Lyman, Balmer, etc.)
    '''
    def __init__(self):
        self.rdb = reactions_database()

        # add reactions selection for Balmer lines
        self.RR = {
            'balmer': ["AMJUEL,12,2_1_5a",  # H(3)/H 
                       "AMJUEL,12,2_1_8a",  # H(3)/H+ 
                       "AMJUEL,12,2_2_5a",  # H(3)/H2
                       "AMJUEL,12,2_2_14a", # H(3)/H2+
                       "AMJUEL,12,2_0c",    # H2+/H2
                       "AMJUEL,12,7_2a",    # H(3)/H-
                       "AMJUEL,11,7_0a",    # H-/H2
                       "AMJUEL,12,2_2_15a", # H(3)/Hll3+
                       "AMJUEL,11,4_0a"     # H3+/H2/H2+/ne
            ],
            'lyman': [ ], # TODO
            'paschen': [ ], # TODO
        }
        
        # spontaneous emission coeffs for n=2 to 1, 3 to 1, ... 16 to 1
        self.A_vals = {
            'lyman': [4.699e8, 5.575e7, 1.278e7, 4.125e6, 1.644e6, 7.568e5, 3.869e5, 2.143e5,
                      1.263e5, 7.834e4, 5.066e4, 3.393e4, 2.341e4, 1.657e4, 1.200e4],
            'balmer': [4.41e7, 8.42e6, 2.53e6, 9.732e5, 4.389e5, 2.215e5, 1.216e5, 7.122e4,
                       4.397e4, 2.83e4, 18288.8, 12249.1, 8451.26, 5981.95, 4332.13],
            'paschen': [] # TODO                       
        }
        
    def load_pec(self, ne, Te, ni, nh, nh2,
                 series='balmer', choice='alpha', plot=False):
        '''Read PECs for a chosen H-isotope line.

        Parameters
        ----------
        ne_m3 : float or N-dim array
            Electron density in units of :math:`m^{-3}`.
        Te_m3 : float or N-dim array
            Electron temperature in units of :math:`eV`.
        ni_m3 : float or N-dim array
            Proton density in units of :math:`m^{-3}`.
        nh_m3 : float or N-dim array
            Neutral atomic hydrogen density in units of :math:`m^{-3}`.
        nh2_m3 : float or N-dim array
            Neutral molecular hydrogen density in units of :math:`m^{-3}`.
        series : str
            Name of the H series of interest, one of ['lyman','balmer',...]
        choice : str
            Selection among ['alpha','beta','gamma','delta']
        plot : bool
            If True, display the selected rates.

        Returns
        -------
        c1 : array
            Contribution to emissivity from atomic neutral H
        c2 : array
            Contribution to emissivity from protons (H+)
        c3 : array
            Contribution to emissivity from molecular neutral H2
        c4 : array
            Contribution to emissivity from molecular ionized H2 (H2+)
        c5 : array
            Contribution to emissivity from atomic neutral charged H2 (H2-)
        '''
        ins = {'ne': np.atleast_1d(ne),
               'Te': np.atleast_1d(Te),
               'ni': np.atleast_1d(ni),
               'nh': np.atleast_1d(nh),
               'nh2': np.atleast_1d(nh2)}
        for key in ins:
            try:
                assert ins[key].shape == ins['ne'].shape
            except:
                raise ValueError(f'Input shape of {key} array is not the same as the ne array!')
        
        # changing final reaction letter
        sub = {"alpha": "a", "beta": "c", "gamma": "d", "delta": "e"}

        RR = copy.deepcopy(self.RR[series])
        for i in [0, 1, 2, 3, 5, 7]:  # only cross sections specific to Halpha
            RR[i] = "AMJUEL" + RR[i].split("AMJUEL")[1].replace("a", sub[choice])
        
        # select Einstein Aki coefficients
        ind = {'alpha': 0, 'beta': 1, 'gamma': 2, 'delta': 3}
        a0 = self.A_vals[series][ind[choice]]

        # calculate rates
        rates = np.zeros((len(RR), *ins['ne'].shape))
        for j, R in enumerate(RR):
            self.rdb.select_reaction(R)
            rates[j, :] = self.rdb.reaction(ins['ne'], ins['Te'])

        c1 = a0 * rates[0, :] * ins['nh']
        c2 = a0 * rates[1, :] * ins['ni']
        c3 = a0 * rates[2, :] * ins['nh2']
        c4 = a0 * rates[3, :] * rates[4, :] * ins['nh2']
        c5 = a0 * rates[5, :] * rates[6, :] * ins['nh2']
        ct = c1 + c2 + c3 + c4 + c5

        if plot:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(ins['Te'], ct)
            ax[1].plot(ins['Te'], c1 / ct, label="H")
            ax[1].plot(ins['Te'], c2 / ct, label="H+")
            ax[1].plot(ins['Te'], c3 / ct, label="H2")
            ax[1].plot(ins['Te'], c4 / ct, label="H2+")
            ax[1].plot(ins['Te'], c5 / ct, label="H2-")
            ax[1].legend()
            ax[1].set_yscale("log")
            ax[1].set_ylim((1e-3, 1e0))
            ax[1].set_xlabel("$T_e$ [eV]")
            ax[0].set_ylabel("$\epsilon_{tot}$ [ph/m$^{-3}$]")
            ax[1].set_ylabel("$\epsilon_i/\epsilon_{tot}$")

        return c1, c2, c3, c4, c5


def extract_data_block(s, lower=True, repl=[]):
    for af in [
        "An analytic formula is given in the text.",
        "See text for analytic formulas.",
    ]:
        if af in s:
            return None

    if (s.count(r"\begin{verbatim}") != 1) or (s.count(r"\end{verbatim}") != 1):
        return None
    block = s.split(r"\begin{verbatim}")[1].split(r"\end{verbatim}")[0]
    if lower:
        block = block.lower()
    for r1, r2 in repl:
        block = block.replace(r1, r2)
    return block


def read_coefficients1D(block, varname="a", fmt="%.12e"):
    D = np.zeros(10) * np.nan
    for i in range(10):
        sc = "%s%i " % (varname, i)
        if block.count(sc) == 1:
            D[i] = float(block.split(sc)[1].split()[0])
    if ~np.isnan(D[9]):
        print("unexpected data block length")
        return []

    D = D[0:9]
    if np.any(np.isnan(D)):
        print("unrecognized data:")
        print(block)
        return []

    return D.tolist()


def read_coefficients2D(block, fmt="%.12e"):
    T = block.replace("t index", "t-index:").split("t-index:")[1:]
    s = "\n"
    for i in range(9):
        for j in range(3):
            l = T[j].split("\n")[i + 1]
            s += ",".join(l.split()[1:]) + ","
        s += "\n"

    tmp = np.fromstring(s.replace("d", "e"), count=9 * 9, sep=",", dtype=float)
    return tmp.reshape(9, 9).tolist()


def read_variables(block, reac, vnfd):
    for v, n, f, d in vnfd:
        if not (d is None):
            reac[n] = d
        if block.count(v) == 1:
            x = float(block.split(v)[1].split()[0]) * f
            reac[n] = x


def parse_hydhel():
    """Download and parse HYDHEL database.

    Sections:
    H1: sigma(E)
    H2: <sigma v>(T)
    H3: <sigma v>(E,T)
    H8: <sigma v E>(T)

    Returns
    -------
    database : dict
        Dictionary containing parse HYDHEL reactions.
    """
    # download and store tex file to disk
    link = "http://www.eirene.de/hydhel.tex"
    r = requests.get(link)
    with open(local_path+os.sep+"hydhel.tex", "wb") as f:
        f.write(r.content)

    # now read the tex file back in for consistency
    hh=open(local_path+os.sep+'hydhel.tex').read()

    database = {}
    report = "HYDHEL"

    headers = {}
    for h in hh.split("\section{H.")[1:]:
        headers[int(h[0:2])] = h

    for ih in [1, 2, 8]:
        for S in headers[ih].split("\subsection{"):
            # ll=S.split('\r\n')
            ll = S.split("\n")
            if ll[1].startswith("Reaction"):
                nam = ll[1].split()[1]
                latex = ll[1].split("$")[1]

                reaction = {"report": report, "header": ih, "name": nam, "latex": latex}

                reaction["symbol"] = {
                    1: r"$\sigma$",
                    2: r"$\langle\sigma\cdot v\rangle$",
                    8: r"$\langle\sigma\cdot v\cdot E\rangle$",
                }[ih]
                reaction["unit"] = {
                    1: "m$^2$",
                    2: "m$^3$ s$^{-1}$",
                    8: "m$^3$ eV s$^{-1}$",
                }[ih]
                reaction["parameters"] = {1: "E", 2: "E,T", 8: "T"}[ih]
                reaction["factor"] = {1: 1e-4, 2: 1e-6, 8: 1e-6}[ih]

                block = extract_data_block(S, repl=[[",", " "]])
                if not (block is None):
                    reaction["coefficients"] = read_coefficients1D(
                        block, varname={1: "a", 2: "b", 8: "h"}[ih]
                    )
                    read_variables(block, reaction, vnfdEnergy)

                database[
                    "%s,%i,%s" % (report, ih, nam.replace(".", "_"))
                ] = reaction  # the symbol . is used by json/fson to address a child element

    for ih in [3]:
        for S in headers[ih].split("\subsection{"):
            # ll=S.split('\r\n')
            ll = S.split("\n")
            if ll[1].startswith("Reaction"):
                nam = ll[1].split()[1]
                latex = ll[1].split("$")[1]

                reaction = {
                    "report": report,
                    "header": ih,
                    "name": nam,
                    "latex": latex,
                    "symbol": r"$\langle\sigma\cdot v\rangle$",
                    "unit": "m$^3$ s$^{-1}$",
                    "parameters": "E,T",
                    "factor": 1.0e-6,
                }

                block = extract_data_block(S, repl=[[",", " "]])
                if not (block is None):
                    reaction["coefficients"] = read_coefficients2D(block)
                    read_variables(block, reaction, vnfdEnergy)

                database[
                    "%s,%i,%s" % (report, ih, nam.replace(".", "_"))
                ] = reaction  # the symbol . is used by json/fson to address a child element

    return database


def parse_amjuel():
    """Download and parse AMJUEL database.

    Sections:
    H1: sigma(E)
    H2: <sigma v>(T)
    H3: <sigma v>(E,T)
    H4: <sigma v>(n,T)
    H5: not in use
    H6: <sigma v p>(E,T)
    H7: <sigma v>(E,T)
    H8: <sigma v E>(T)
    H9: <sigma v E>(E,T)
    H10: <sigma v E>(E,T)
    H11: misc - other single polynomial fits
    H12: misc - other double polynomial fits

    Returns
    -------
    database : dict
        Dictionary containing parse AMJUEL reactions.
    """
    database = {}

    # first download the databases
    link = "http://www.eirene.de/amjuel.tex"
    r = requests.get(link)
    with open(local_path+os.sep+"amjuel.tex", "wb") as f:
        f.write(r.content)

    report = "AMJUEL"

    # open file back
    aj=open(local_path+os.sep+'amjuel.tex').read()

    headers = aj.split("\section{H.")[1:]

    for ih in [3, 4, 10, 12]:
        header = headers[ih]

        for r in header.split("\subsection{")[1:]:
            l3 = (" ".join(r.split("\n")[0:3])).strip()
            nam = l3.split()[1]

            latex = l3.split("$")[1].strip()
            reaction = {"report": report, "header": ih, "name": nam, "latex": latex}

            reaction["symbol"] = {
                3: r"$\langle\sigma\cdot v\rangle$",
                4: r"$\langle\sigma\cdot v\rangle$",
                10: r"$\langle\sigma\cdot v\cdot E\rangle$",
                12: "ratio",
            }[ih]
            reaction["unit"] = {
                3: "m$^3$ s$^{-1}$",
                4: "m$^3$ s$^{-1}$",
                10: "m$^3$ eV s$^{-1}$",
                12: "",
            }[ih]
            reaction["parameters"] = {3: "E,T", 4: "n,T", 10: "n,T", 12: "n,T"}[ih]
            reaction["factor"] = {3: 1e-6, 4: 1e-6, 10: 1e-6, 12: 1}[ih]
            reaction["factor_n"] = {3: 1.0, 4: 1e-14, 10: 1e-14, 12: 1e-14}[ih]

            block = ""
            data = ""
            block = extract_data_block(r)

            if not (block is None):
                reaction["coefficients"] = read_coefficients2D(block)
                read_variables(block, reaction, vnfdTn)

            # the symbol . is used by json/fson to address a child element
            database[
                "%s,%i,%s" % (report, ih, nam.replace(".", "_"))
            ] = reaction

    for ih in [2, 8, 11]:
        for S in headers[ih].split("\subsection{")[1:]:
            ll = S.split("\n")
            if ll[1].startswith("Reaction"):
                nam = ll[1].split()[1]
                latex = (ll[1] + ll[2]).split("$")[1]
                reaction = {
                    "report": report,
                    "header": ih,
                    "name": nam,
                    "latex": latex,
                    "parameters": "T",
                }

                reaction["symbol"] = {
                    2: r"$\langle\sigma\cdot v\rangle$",
                    8: r"$\langle\sigma\cdot v\cdot E\rangle$",
                    11: r"$\langle E\rangle$",
                }[ih]
                reaction["unit"] = {
                    2: "m$^3$ s$^{-1}$",
                    8: "m$^3$ eV s$^{-1}$",
                    11: "eV",
                }[ih]
                reaction["factor"] = 1 if ih == 11 else 1e-6

                block = extract_data_block(S, repl=[[",", " "], ["d", "e"]])

                if not (block is None):
                    reaction["coefficients"] = read_coefficients1D(
                        block, varname={1: "a", 2: "b", 8: "h", 11: "k"}[ih]
                    )
                    read_variables(block, reaction, vnfdEnergy)

                # the symbol . is used by json/fson to address a child element
                database[
                    "%s,%i,%s" % (report, ih, nam.replace(".", "_"))
                ] = reaction

    return database


