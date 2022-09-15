import aurora
import matplotlib.pyplot as plt
plt.ion()

shot = 38996
time_s = 3.5
label='osm_test'
casename = f'oedge_AUG_{shot}_{int(time_s*1e3)}_{label}'

# set up an OEDGE case
osm = aurora.oedge_case(shot, (t0+t1)/2., label=label)

# load a specific input file
osm.load_input_file(filepath='/path/to/my/file.d6i')

# possibly modify specific parts of the input file:
osm.inputs['+P01']['data'] = 22

# update input file
osm.write_input_file(filepath='/path/to/my/file.d6i')

# now run simulation
osm.run(grid_loc='/path/to/my/grid/file.sno')

# once simulation is run, load the output
osm.load_output()

# now, initialize emission calculation
am = aurora.h_am_pecs()

# extract relevant fields from the OEDGE run
ne_m3 = osm.output.read_data_2d('KNBS')
Te_eV = osm.output.read_data_2d('KTEBS')
Ti_eV = osm.output.read_data_2d('KTIBS')
n_H2_m3 = osm.output.read_data_2d('PINMOL')
n_H_m3 = osm.output.read_data_2d('PINATO')

# load all contributions predicted by AMJUEL
cH,cHp,cH2,cH2p,cH2m = am.load_pec(
    ne_m3, Te_eV, ne_m3, # ni=ne
    n_H_m3, n_H2_m3,
    series='balmer', choice='alpha', plot=False
)

fig,axs = plt.subplots(1,5, figsize=(25,6), sharex=True)
osm.output.plot_2d(cH, ax=axs[0]); axs[0].set_title('cH')
osm.output.plot_2d(cHp, ax=axs[1]); axs[1].set_title('cHp')
osm.output.plot_2d(cH2, ax=axs[2]); axs[2].set_title('cH2')
osm.output.plot_2d(cH2p, ax=axs[3]); axs[3].set_title('cH2p')
osm.output.plot_2d(cH2m, ax=axs[4]); axs[4].set_title('cH2m')
plt.tight_layout()
