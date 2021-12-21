import yt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.offsetbox import AnchoredText
from yt.units import Msun, pc, kpc, Mpc, km, s, Gyr, Kelvin, g, cm, Myr, erg
from numpy import sqrt, pi
from yt.utilities.physical_constants import mp, G, kb, c

draw_density_map        = 0
draw_temperature_map    = 0
draw_SFR                = 0
draw_PDF                = 0
draw_mbh_growth         = 0
draw_mbh_position       = 0
draw_gas_mass           = 0
draw_gas_outflow        = 1


axis                    = ['x', 'z']
times                   = [500] #[0, 500] # in Myr
figure_width            = 30 # in kpc

codes                   = ['ENZO']
marker_names             = ['o']

filenames = [
    # "/data/shared/AGORA/IsolatedAGN/1e8_seed/DD????/DD????",         # seeding
    # "/data/shared/AGORA/IsolatedAGN/1e8/DD????/DD????",          # Bondi-Hoyle accretion
    # "/data/shared/AGORA/IsolatedAGN/1e8_thermal/DD????/DD????",  # Bondi-Hoyle feedback
    "/data/shared/AGORA/IsolatedAGN/acc1/DD????/DD????",           # constant accretion
    "/data/shared/AGORA/IsolatedAGN/acc1_thermal/DD????/DD????",   # constant feedback
]
labels = [
    # "seed",
    # "BH_accretion",
    # "BH_thermalFB",
    "const_accretion",
    "const_thermalFB"
]
dir_plot="IMG_isolatedAGN_paper"

fig_density_map         = []
fig_temperature_map     = []
fig_PDF                 = []
grid_density_map        = []
grid_temperature_map    = []
grid_PDF                = []
sfr_ts                  = []
sfr_sfrs                = []
mbh_rate_ts             = []
mbh_rate_mass           = []
mbh_rate                = []
mbh_rate_edd            = []
mbh_position_ts         = []
mbh_position            = []
gas_mass_xs             = []
gas_mass_profiles       = []
gas_outflow_ts          = []
gas_outflow_profiles    = []


for code in range(len(codes)):
    if draw_mbh_growth == 1:
        mbh_rate_ts.append([])
        mbh_rate_mass.append([])
        mbh_rate.append([])
        mbh_rate_edd.append([])
        
    if draw_gas_mass == 1:
        gas_mass_xs.append([])
        gas_mass_profiles.append([])

    if draw_mbh_position == 1:
        mbh_position_ts.append([])
        mbh_position.append([])
    
    if draw_SFR == 1:
        sfr_ts.append([])
        sfr_sfrs.append([])
    
    if draw_gas_outflow == 1:
        gas_outflow_ts.append([])
        gas_outflow_profiles.append([])
        
    
for t, time in enumerate(times):
    if draw_density_map == 1:
        fig_density_map     += [plt.figure(figsize=(5*len(codes), 5*len(axis)))]
        grid_density_map    += [AxesGrid(fig_density_map[t],
                                    (0.01, 0.01, 0.99, 0.99),
                                    nrows_ncols = (len(axis), len(codes)),
                                    axes_pad = 0.02,
                                    add_all = True,
                                    share_all = True,
                                    label_mode = "1",
                                    cbar_mode = "single",
                                    cbar_location = "right",
                                    cbar_size = "2%",
                                    cbar_pad = 0.02)]
        
    if draw_temperature_map == 1:
        fig_temperature_map  += [plt.figure(figsize=(5*len(codes), 5*len(axis)))]
        grid_temperature_map += [AxesGrid(fig_temperature_map[t],
                                    (0.01, 0.01, 0.99, 0.99),
                                    nrows_ncols = (len(axis), len(codes)),
                                    axes_pad = 0.02,
                                    add_all = True,
                                    share_all = True,
                                    label_mode = "1",
                                    cbar_mode = "single",
                                    cbar_location = "right",
                                    cbar_size = "2%",
                                    cbar_pad = 0.02)]
    if draw_PDF == 1:
        fig_PDF              += [plt.figure(figsize=(7*len(codes), 7))]
        grid_PDF             += [AxesGrid(fig_PDF[t],
                                    (0.01, 0.01, 0.99, 0.99),
                                    nrows_ncols = (1, len(codes)),
                                    axes_pad = 0.05,
                                    add_all = True,
                                    share_all = True,
                                    label_mode = "1",
                                    cbar_mode = "single",
                                    cbar_location = "right",
                                    cbar_size = "2%",
                                    cbar_pad = 0.05,
                                    aspect=False)]

##########################       
##    PARTICLE FILTER
##########################  
def _MBH_filter(pfilter, data):                        
    return data[("all", "particle_type")] == 8
yt.add_particle_filter("MBH", function = _MBH_filter, filtered_type='all', requires=["particle_type"])

def _NewStar_filter(pfilter, data):                        
    return np.logical_and( data[("all", "particle_type")] == 2, data[("all", "creation_time")] > 0 )
yt.add_particle_filter("newstar", function = _NewStar_filter, 
                       filtered_type='all', requires=["particle_type","creation_time"])

##########################       
##    ANALYSIS
##########################  
def load_file(fn, time):
    ts=yt.load(fn)
    for ds in ts:
        current_time = ds.current_time.in_units("Myr").round()
        if time == current_time:
            return ds
        
    return ts[-1]

def get_figure_center(ds):
    
     # Find mbh center
    ds.add_particle_filter("MBH")        
    ad = ds.all_data()
    bhx = ad["MBH","particle_position_x"]
    bhy = ad["MBH","particle_position_y"]
    bhz = ad["MBH","particle_position_z"]

    if len(bhx) > 0:
        center = [bhx.d[0], bhy.d[0], bhz.d[0]]
    else : 
        center=[0.5,0.5,0.5]
        rad= 30
        while rad > 1:
            sp=ds.sphere(center=center, radius=(rad, 'kpc'))
            center=sp.quantities.center_of_mass(use_gas=True, use_particles=False)
            rad=rad*0.5
    return center

def get_mbh_growth(fn, initial_mass=1e8):
    ts=yt.load(fn)
    
    sigma_thompson = 6.6524e-25*cm**2
    current_mass = initial_mass
    current_time = 0
    
    mbh_rate_mass = [initial_mass]
    mbh_rate = []
    mbh_rate_edd =[]
    mbh_rate_ts = [0]
    
    for ds in ts[1:]:
        time_tmp = int(ds.current_time.in_units("Myr").round())
        ds.add_particle_filter("MBH")
        ad = ds.all_data()
        bhmass = ad["MBH","particle_mass"].in_units("Msun")
        
        if len(bhmass)==0:
            break
        
        mbh_rate_ts.append(current_time)
        mbh_rate_mass.append(bhmass.d[0])
        
        ## accretion rate
        Medd = ds.quan((current_mass+bhmass.d[0])/2,"Msun").in_units("g")
        Ledd = (4*pi*G*Medd*mp*c/sigma_thompson).in_units("erg/s")
        Mdot_edd = (Ledd/(0.1*c**2)).in_units("Msun/yr")
        diff = bhmass.d[0] - current_mass
        diff_t = time_tmp-current_time
        current_mass = bhmass.d[0]
        current_time = time_tmp

        mbh_rate.append(diff/diff_t/1e6) #Myr to yr
        mbh_rate_edd.append(diff/diff_t/1e6/Mdot_edd)       
    
    return mbh_rate_ts, mbh_rate_mass, mbh_rate, mbh_rate_edd

def get_mbh_position(fn):
    ts=yt.load(fn)
    mbh_position_ts=[]
    mbh_position=[]
    for ds in ts[1:]:
        ## time
        current_time = int(ds.current_time.in_units("Myr").round())
        mbh_position_ts.append(current_time)
        
        ## MBH position
        ds.add_particle_filter("MBH")
        ad = ds.all_data() 
        bhpos = ad["MBH","particle_position"].in_units('kpc')
        
        ## Center of Mass
        center=[0.5,0.5,0.5]
        rad= 30
        while rad > 1:
            sp=ds.sphere(center=center, radius=(rad, 'kpc'))
            center=sp.quantities.center_of_mass(use_gas=True, use_particles=False)
            rad=rad*0.5
        center=center.in_units('kpc')
        
        ## Distance
        dx = 0
        for x in range(3):
            dx += (bhpos.d[0][x]-center.d[x])**2
        mbh_position.append(np.sqrt(dx))
    
    return mbh_position_ts, mbh_position

def get_sfr(fn, nbins=25):
    
    ts=yt.load(fn)
    ds=ts[-1]
    ds.add_particle_filter("newstar")
    ad= ds.all_data()
    smass = ad[("newstar", "particle_mass")].in_units("Msun")
    sct = ad[("newstar", "creation_time")].in_units("Myr")
    sage = ad[("newstar", "age")].in_units("Myr")

    # current time
    current_time = ds.current_time.in_units("Myr")
    f_eject = 0.163

    # Initial Mass calculation using star age
    smass_initial = []
    for j in range(len(sage.d)):
        sage_p = sage.d[j]
        smass_p = smass.d[j]
        if sage_p > 12 :
            sage_p=12
        smass_i = smass_p/(1-f_eject*(1-(1+sage_p)*np.exp(-sage_p)))
        smass_initial.append(smass_i)
    smass_initial = np.array(smass_initial)

    # Binning times
    tmin = sct.min()
    time_bins = np.linspace(tmin*1.01, current_time, nbins+1)
    
    # Binning star initial mass
    inds = np.digitize(sct, time_bins)-1
    mass_bins = np.zeros(nbins+1, dtype='float64')*Msun
    for index in np.unique(inds):
        mass_bins[index] += smass_initial[inds == index].sum()
        
    # Calculate StarFormationRate [Msun/yr]
    time_bins_dt = time_bins[1:] - time_bins[:-1]
    sfr = (mass_bins[:-1] / time_bins_dt).in_units('Msun/yr')
    sfr_time = 0.5*(time_bins[1:]+time_bins[:-1]).in_units('Myr')

    return sfr_time, sfr


def get_gas_outflow(fn, figure_width, height, nbins=30):
    
    ts=yt.load(fn)
    gas_ts=[]
    gas_outflow=[]
    
    for h_tmp in height:
        gas_outflow.append([])

    for ds in ts[1:]:
        ## time
        current_time = int(ds.current_time.in_units("Myr").round())
        gas_ts.append(current_time)
        
        ## data object : shallow cylinder
        center=get_figure_center(ds)
        sp= ds.disk(center, [0.,0.,1.], (0.3,'kpc'), (15,'kpc'))
        
        ## profile plot of gas z-velocity
        p = yt.ProfilePlot(sp, ("index", "cylindrical_z"), ("gas", "velocity_cylindrical_z"), weight_field=("gas", "cell_mass"), n_bins=nbins, x_log=False, accumulation=False)
        p.set_log("velocity_cylindrical_z", False)
        p.set_log("cylindrical_z", False)
        p.set_unit("cylindrical_z", 'kpc')
        p.set_xlim(1e-3, 0.5*figure_width)
        hgt = p.profiles[0].x.in_units('kpc').d
        velocity = p.profiles[0]["velocity_cylindrical_z"].in_units('km/s').d
        
        ## Select gas outflow at specific height
        inds = np.digitize(height, hgt)
        for x, index in enumerate(inds):
            gas_outflow[x].append(velocity[index])
    
    return gas_ts, gas_outflow

    

############################################################################################################

for fn, label in zip(filenames, labels):
    
    for code in range(len(codes)):
        
        ##################################################
        ##    Properties over time
        ##################################################

        if draw_mbh_growth == 1 and time != 0:
            mbh_ts_tmp, mbh_mass_tmp, mbh_rate_tmp, mbh_rate_edd_tmp=get_mbh_growth(fn, initial_mass=1e8)
            mbh_rate_ts[code]=mbh_ts_tmp
            mbh_rate_mass[code]=mbh_mass_tmp
            mbh_rate[code]=mbh_rate_tmp
            mbh_rate_edd[code]=mbh_rate_edd_tmp


        if draw_mbh_position == 1 and time != 0:
            mbh_ts_tmp, mbh_position_tmp = get_mbh_position(fn)
            mbh_position_ts[code]=mbh_ts_tmp
            mbh_position[code]=mbh_position_tmp


        if draw_SFR == 1 and time != 0:
            times_tmp, sfr_tmp = get_sfr(fn, nbins=25)
            sfr_ts[code]=times_tmp
            sfr_sfrs[code]=sfr_tmp

        if draw_gas_outflow == 1:
            height = [3, 10] #kpc
            times_tmp, vel_tmp = get_gas_outflow(fn, figure_width, height)
            gas_outflow_ts[code].append(times_tmp)
            gas_outflow_profiles[code].append(vel_tmp)
            
    
        ##################################################
        ##    Snapshot at specific time step
        ##################################################
        for t, time in enumerate(times):
            # Load dataset
            ds = load_file(fn, time)

            # Get center
            center = get_figure_center(ds)
            ## Density Map
            if draw_density_map == 1:
                for x, ax in enumerate(axis):
                    p = yt.ProjectionPlot(ds, ax, ("gas", "density"), center=center, width=(figure_width, 'kpc'), weight_field=None, fontsize=20)
                    p.set_zlim(("gas", "density"), 1e-5, 1e-1)
                    p.set_cmap(("gas", "density"), 'viridis')
                    plot = p.plots[("gas", "density")]
                    plot.figure = fig_density_map[t]
                    plot.axes = grid_density_map[t][x*len(codes)+code]
                    if code == 0:
                        plot.cax = grid_density_map[t].cbar_axes[0]
                    p._setup_plots()

                    at = AnchoredText(codes[code], loc=2, prop=dict(size=18), frameon=True)
                    grid_density_map[t][code].axes.add_artist(at)

            ## Temperature Map
            if draw_temperature_map == 1:
                for x, ax in enumerate(axis):
                    p1 = yt.SlicePlot(ds, ax, ("gas", "temperature"), center=center, width=(figure_width, 'kpc'), fontsize=20)
                    p1.set_zlim(("gas", "temperature"), 1e2, 1e7)
                    p1.set_cmap(("gas", "temperature"), 'dusk')
                    plot1 = p1.plots[("gas", "temperature")]
                    plot1.figure = fig_temperature_map[t]
                    plot1.axes = grid_temperature_map[t][x*len(codes)+code]
                    if code == 0:
                        plot1.cax = grid_temperature_map[t].cbar_axes[0]
                    p1._setup_plots()

                    at = AnchoredText(codes[code], loc=2, prop=dict(size=18), frameon=True)
                    grid_temperature_map[t][code].axes.add_artist(at)

            ## Density-Temperature PDF
            if draw_PDF == 1:
                sp = ds.sphere(center, (0.5*figure_width, "kpc"))
                p2 = yt.PhasePlot(sp, ("gas", "density"), ("gas", "temperature"), ("gas","cell_mass"), weight_field=None, fontsize=20, x_bins=300, y_bins=300)
                p2.set_unit("cell_mass", "Msun")
                p2.set_xlim(1e-33, 6e-22)
                p2.set_ylim(20, 2e8)
                p2.set_zlim(("gas", "cell_mass"), 1e2, 1e8)
                p2.set_cmap(("gas", "cell_mass"), 'viridis')
                p2.set_colorbar_label(("gas", "cell_mass"), "Mass ($\mathrm{M}_{\odot}$)")
                plot2 = p2.plots[("gas", "cell_mass")]

                plot2.figure = fig_PDF[t]
                plot2.axes = grid_PDF[t][code].axes
                if code ==0:
                    plot2.cax = grid_PDF[t].cbar_axes[0]
                p2._setup_plots()

                at = AnchoredText(codes[code], loc=3, prop=dict(size=18), frameon=True)
                grid_PDF[t][code].axes.add_artist(at)  

            ## Gas mass profile
            if draw_gas_mass == 1 and time != 0:
                sp = ds.sphere(center, (0.5*figure_width, "kpc"))
                sp.set_field_parameter("normal", [0.,0.,1.]) 
                p3 = yt.ProfilePlot(sp, ("index", "cylindrical_r"), ("gas", "cell_mass"), weight_field=None, n_bins=50, x_log=False, accumulation=False)
                p3.set_log("cell_mass", True)
                p3.set_log("cylindrical_r", False)
                p3.set_unit("cylindrical_r", 'kpc')
                p3.set_xlim(1e-3, 15)

                gas_mass_xs[code]=p3.profiles[0].x.in_units('kpc').d
                gas_mass_profiles[code]=p3.profiles[0]["cell_mass"].in_units('Msun').d
        # end of time loop
    # end of code loop
# end of fn loop


    ##################################################
    ##    Save Figure
    ##################################################        

        # 1. within time loop
        if draw_density_map == 1:
            fig_density_map[t].savefig("{}/{}_Dens_{}Myr".format(dir_plot, label, time), dpi=300, bbox_inches="tight")

        if draw_temperature_map == 1:
            fig_temperature_map[t].savefig("{}/{}_Temp_{}Myr".format(dir_plot,label, time), dpi=300, bbox_inches="tight")
            
        if draw_PDF == 1:
            fig_PDF[t].set_size_inches(7*len(codes),7, forward=True)
            fig_PDF[t].savefig("{}/{}_PDF_{}Myr".format(dir_plot, label, time), dpi=300, bbox_inches="tight")

        if draw_gas_mass == 1 and time != 0:
            plt.clf()
            fig = plt.figure(figsize=(8,6))
            for code in range(len(codes)):
                lines = plt.plot(gas_mass_xs[code], np.add.accumulate(gas_mass_profiles[code]), linewidth=1.2, alpha=0.8)
            plt.semilogy()
            plt.xlim(0, 14)
            plt.ylim(1e7, 2e10)
            plt.grid(True)
            plt.xlabel("$\mathrm{Cylindrical\ Radius\ (kpc)}$", fontsize='large')
            plt.ylabel("$\mathrm{Mass\ (M_{\odot})}$", fontsize='large')
            plt.legend(codes, loc=4, frameon=True, ncol=2, fancybox=True)
            leg = plt.gca().get_legend()
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize='small')
            plt.savefig("{}/{}_Gas_mass_{}Myr".format(dir_plot, label, time), dpi=300, bbox_inches='tight')
            
            
    # 2. outside time loop
    if draw_mbh_growth == 1 and time != 0:
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        for code in range(len(codes)):
            lines = plt.plot(mbh_rate_ts[code][1:], mbh_rate[code], linewidth=1.2, alpha=0.8)
        plt.xlim(0, time)
        plt.ylim(5e-6, 20)
        plt.grid(True)
        plt.xlabel("$\mathrm{Time\ (Myr)}$", fontsize='large')
        plt.ylabel("$\mathrm{MBH\ Accretion\ Rate\ (M_{\odot}/yr)}$", fontsize='large')
        plt.yscale("log")
        plt.legend(codes, loc=2, frameon=True, ncol=2, fancybox=True)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize='small')
        plt.savefig("{}/{}_MBH_Accretion_Rate_{}Myr".format(dir_plot, label, time), dpi=300, bbox_inches='tight')
        plt.clf()

    if draw_mbh_position == 1 and time != 0:
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        for code in range(len(codes)):
            lines = plt.plot(mbh_position_ts[code], mbh_position[code], marker=marker_names[code], linewidth=1.2, alpha=0.8)
        plt.xlim(0, time)
        plt.ylim(-0.02, 2)
        plt.grid(True)
        plt.xlabel("$\mathrm{Time\ (Myr)}$", fontsize='large')
        plt.ylabel("$\mathrm{Distance\ (kpc)}$", fontsize='large')
        # plt.yscale("log")
        plt.legend(codes, loc=2, frameon=True, ncol=2, fancybox=True)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize='small')
        plt.savefig("{}/{}_MBH_Distance_{}Myr".format(dir_plot, label, time), dpi=300, bbox_inches='tight')
        plt.clf()



    if draw_SFR == 1 and time != 0:
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        for code in range(len(codes)):
            lines = plt.plot(sfr_ts[code], sfr_sfrs[code], marker=marker_names[code], linewidth=1.2, alpha=0.8)
        plt.xlim(0, time)
        plt.ylim(0, 25)
        plt.grid(True)
        plt.xlabel("$\mathrm{Time\ (Myr)}$", fontsize='large')
        plt.ylabel("$\mathrm{Star\ Formation\ Rate\ (M_{\odot}/yr)}$", fontsize='large')
        plt.legend(codes, loc=4, frameon=True, ncol=2, fancybox=True)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize='small')
        plt.savefig("{}/{}_SFR_{}Myr".format(dir_plot, label, time), dpi=300, bbox_inches='tight')
        plt.clf()
            
# 3. outside fn loop
if draw_gas_outflow == 1 and time != 0:
    plt.clf()
    fig, ax = plt.subplots(1, len(height), figsize=(8*len(height), 6))
    axs = ax.ravel()

    for x, axis in enumerate(axs):

        for l, label in enumerate(labels):
            if 'accretion' in label:
                linestyle='--'
                linecolor='tab:blue'
                lab='accretion'
            else:
                linestyle='-'
                linecolor='tab:blue'
                lab='feedback'
            axis.plot(gas_outflow_ts[code][l], gas_outflow_profiles[code][l][x], linewidth=1.2, alpha=0.8, linestyle=linestyle, color=linecolor, label="{}_{}".format(codes[code], lab))
    
        axis.grid(True)
        axis.set_ylim(-500, 4500)
        axis.set_xlabel("$\mathrm{Time\ (Myr)}$", fontsize='large')
        axis.set_ylabel("$\mathrm{Gas\ z-Velocity\ \ (km/s)}$", fontsize='large')
        axis.set_title("Z height = {} kpc".format(height[x]))
        axis.legend(loc=1, frameon=True, ncol=1, fancybox=True)    

    plt.savefig("{}/const_GasFlow_{}Myr".format(dir_plot, time), dpi=300, bbox_inches='tight')
    plt.clf()