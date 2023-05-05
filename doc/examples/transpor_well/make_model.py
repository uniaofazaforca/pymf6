"""Create, run, and postprocess a MODFLOW 6 model with flopy.

Model:Flow and transport model with one well
"""

import flopy
from flopy.utils.postprocessing import get_specific_discharge
from matplotlib import pyplot as plt
import numpy as np
import os

# function to create the model: all the packages
def make_input(
    qwell,
    wel_coords,
    cwell,
    model_path,
    name,
    mixelm, # advection scheme 0 = upstream , -1 = TVD
    exe_name='mf6',
    verbosity_level=0):
    """Create MODFLOW 6 input"""

    length_units = "feet"
    time_units = "days"

    nlay = 1  # Number of layers
    nrow = 31  # Number of rows
    ncol = 31  # Number of columns
    delr = 900.0  # Column width ($ft$)
    delc = 900.0  # Row width ($ft$)
    delz = 20.0  # Layer thickness ($ft$)
    top = 0.0  # Top of the model ($ft$)
    prsity = 0.35  # Porosity
    dum1 = 2.5  # Length of the injection period ($years$)
    dum2 = 7.5  # Length of the extraction period ($years$)
    k11 = 432.0  # Horizontal hydraulic conductivity ($ft/d$)
    #qwell = 1.0  # Volumetric injection rate ($ft^3/d$)
    #cwell = 100.0  # Relative concentration of injected water ($\%$)
    al = 100.0  # Longitudinal dispersivity ($ft$)
    trpt = 1.0  # Ratio of transverse to longitudinal dispersitivity

    perlen = [912.5, 2737.5]
    nper = len(perlen)
    nstp = [365, 1095]
    tsmult = [1.0, 1.0]
    k11 = (
            0.005 * 86400
    )  # established above, but explicitly writing out its origin here
    sconc = 0.0
    c0 = 0.0
    dt0 = 56.25
    dmcoef = 0
    ath1 = al * trpt
    botm = [top - delz]  # Model geometry
    k33 = k11  # Vertical hydraulic conductivity ($m/d$)
    icelltype = 0
    mixelm = -1
    strt = np.zeros((nlay, nrow, ncol), dtype=float)

    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    icbund = 1

    # MF6 pumping information
    #          (k,  i,  j),  flow, conc
    spd_mf6 = {
        1: [[wel_coords, qwell, cwell]],
        2: [[wel_coords, qwell, cwell]],
        3: [[wel_coords, qwell, cwell]],
        4: [[wel_coords, qwell, 0.0]]
    }

    chdspd = [[(0, 0, 0), 1.],
            [(0, ncol - 1, ncol - 1), 1.]]

    chdspd = {0: chdspd}

    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 1.0
    percel = 1.0  # HMOC parameters
    itrack = 3
    wd = 0.5
    dceps = 1.0e-5
    nplane = 1
    npl = 0
    nph = 16
    npmin = 2
    npmax = 32
    dchmoc = 1.0e-3
    nlsink = nplane
    npsink = nph

    times = (10.0, 120, 1.0)
    tdis_rc = [(1.0, 1, 1.0), times, times, times]

    # MODFLOW 6
    gwfname = "gwf_" + name
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=model_path, exe_name=exe_name
    )

    # Instantiating MODFLOW 6 time discretization
    flopy.mf6.ModflowTdis(
        sim, nper=4, perioddata=tdis_rc, time_units=time_units
    )

    # Instantiating MODFLOW 6 groundwater flow model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        model_nam_file="{}.nam".format(gwfname),
    )

    # Instantiating MODFLOW 6 solver for flow model
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="{}.ims".format(gwfname),
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    # Instantiating MODFLOW 6 discretization package
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        filename="{}.dis".format(gwfname),
    )

    # Instantiating MODFLOW 6 node-property flow package
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=False,
        icelltype=icelltype,
        k=k11,
        k33=k33,
        save_specific_discharge=True,
        filename="{}.npf".format(gwfname),
    )

    # Instantiating MODFLOW 6 storage package (steady flow conditions, so no actual storage, using to print values in .lst file)
    flopy.mf6.ModflowGwfsto(
        gwf, ss=0, sy=0, filename="{}.sto".format(gwfname)
    )

    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(
        gwf, strt=strt, filename="{}.ic".format(gwfname)
    )

    # Instantiating MODFLOW 6 constant head package
    flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=False,
        pname="CHD-1",
        filename="{}.chd".format(gwfname),
    )

    # Instantiate the wel package
    flopy.mf6.ModflowGwfwel(
        gwf,
        print_input=True,
        print_flows=True,
        stress_period_data=spd_mf6,
        save_flows=False,
        auxiliary="CONCENTRATION",
        pname="WEL-1",
        filename="{}.wel".format(gwfname),
    )

    # Instantiating MODFLOW 6 output control package for flow model
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="{}.hds".format(gwfname),
        budget_filerecord="{}.bud".format(gwfname),
        headprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # Instantiating MODFLOW 6 groundwater transport package
    gwtname = "gwt_" + name
    gwt = flopy.mf6.MFModel(
        sim,
        model_type="gwt6",
        modelname=gwtname,
        model_nam_file="{}.nam".format(gwtname),
    )
    gwt.name_file.save_flows = True

    # create iterative model solution and register the gwt model with it
    imsgwt = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="{}.ims".format(gwtname),
    )
    sim.register_ims_package(imsgwt, [gwt.name])

    # Instantiating MODFLOW 6 transport discretization package
    flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        filename="{}.dis".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport initial concentrations
    flopy.mf6.ModflowGwtic(
        gwt, strt=sconc, filename="{}.ic".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport advection package
    if mixelm >= 0:
        scheme = "UPSTREAM"
    elif mixelm == -1:
        scheme = "TVD"
    else:
        raise Exception()
    flopy.mf6.ModflowGwtadv(
        gwt, scheme=scheme, filename="{}.adv".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport dispersion package
    if al != 0:
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=al,
            ath1=ath1,
            filename="{}.dsp".format(gwtname),
        )

    # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
    flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=prsity,
        first_order_decay=False,
        decay=None,
        decay_sorbed=None,
        sorption=None,
        bulk_density=None,
        distcoef=None,
        filename="{}.mst".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]
    flopy.mf6.ModflowGwtssm(
        gwt, sources=sourcerecarray, filename="{}.ssm".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport output control package
    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord="{}.cbc".format(gwtname),
        concentration_filerecord="{}.ucn".format(gwtname),
        concentrationprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    # Instantiate observation package (for transport)
    obslist = [["bckgrnd_cn", "concentration", (0, 15, 15)]]
    obsdict = {"{}.obs.csv".format(gwtname): obslist}
    obs = flopy.mf6.ModflowUtlobs(
        gwt, print_input=False, continuous=obsdict
    )

    # Instantiating MODFLOW 6 flow-transport exchange mechanism
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwtname,
        filename="{}.gwfgwt".format(name),
    )

    sim.write_simulation()
    return sim


def get_simulation(model_path, exe_name='mf6', verbosity_level=0):
    """Get simulation for a model."""
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=model_path,
        exe_name=exe_name,
        verbosity_level=verbosity_level,
    )
    return sim


def run_simulation(model_path, verbosity_level=0):
    """Run a MODFLOW 6 model"""
    sim = get_simulation(
        model_path,
        verbosity_level=verbosity_level)
    sim.run_simulation()


def show_heads(model_path, name):
    """Plot calculated heads along with flow vector."""
    gwfname = "gwf_" + name
    sim = get_simulation(model_path, gwfname)
    gwf = sim.get_model(gwfname)
    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    spdis = bud.get_data(text='DATA-SPDIS')[-1]
    qx, qy, _ = get_specific_discharge(spdis, gwf)
    pmv = flopy.plot.PlotMapView(gwf)
    pmv.plot_array(head)
    pmv.plot_grid(colors='white')
    pmv.contour_array(
        head,
        levels=np.arange(0.2, 1.0, 0.02),
    )
    plot = pmv.plot_vector(
        qx,
        qy,
        normalize=True,
        color="white")
    plot.axes.set_xlabel('X [m]')
    plot.axes.set_ylabel('Y [m]')
    plot.axes.set_title('Head-Controlled Well')
    cbar = plot.get_figure().colorbar(plot)
    cbar.set_label('Water level [m]')
    return plot


def show_well_head(model_path, name, wel_coords):
    """Plot head at well over time."""
    gwfname = "gwf_" + name
    sim = get_simulation(model_path, gwfname)
    gwf = sim.get_model(gwfname)
    heads = gwf.output.head().get_ts(wel_coords)
    _, ax = plt.subplots()
    ax.plot(heads[:, 0], heads[:, 1], label='Well water level')
    ax.set_xlabel('Time [d]')
    ax.set_ylabel('Water level [m]')
    y_start = 0.3
    y_end = 1.05
    y_stress = (y_start, y_end)
    x_stress_1 = (1, 1)
    x_stress_2 = (11, 11)
    x_stress_3 = (21, 21)
    tolerance = 0.01
    head_limit = 0.5
    x = [0, 32]
    ax.set_xlim(*x)
    ax.set_ylim(y_start, y_end)
    y1 = [head_limit - tolerance, head_limit - tolerance]
    y2 = [head_limit + tolerance, head_limit + tolerance]
    ax.plot(x, y1, color='red', linestyle=':', label='Target water level range')
    ax.plot(x, y2, color='red', linestyle=':')
    ax.plot(
         x_stress_1, y_stress,
         color='lightblue', linestyle=':', label='Stress periods')
    ax.plot(
         x_stress_2, y_stress,
         x_stress_3, y_stress,
         color='lightblue', linestyle=':')
    ax.legend(loc=(1.1, 0))
    return ax

def show_concentration(model_path, name, wel_coords):
    """Plot concentrations at well over time."""
    gwtname = "gwt_" + name
    sim = get_simulation(model_path, gwtname)
    gwt = sim.get_model(gwtname)
    conc = gwt.output.concentration().get_ts(wel_coords)
    _, ax = plt.subplots()
    ax.plot(conc[:, 0], conc[:, 1], label='Concentration at the well')
    ax.set_xlabel('Time [d]')
    ax.set_ylabel('Concentration [mg/l]')
    y_start = 0
    y_end = 10
    y_stress = (y_start, y_end)
    x_stress_1 = (1, 1)
    x_stress_2 = (11, 11)
    x_stress_3 = (21, 21)
    x = [0, 32]
    ax.set_xlim(*x)
    ax.set_ylim(y_start, y_end)
    conc_threshold_min = 2
    conc_threshold_max = 4
    y1 = [conc_threshold_min, conc_threshold_min]
    y2 = [conc_threshold_max, conc_threshold_max]
    ax.plot(x, y1,  color='red', linestyle=':', label='Concentration Threshold')
    ax.plot(x, y2, color='red', linestyle=':')
    ax.plot(
        x_stress_1, y_stress,
        color='lightblue', linestyle=':', label='Stress periods')
    ax.plot(
        x_stress_2, y_stress,
        x_stress_3, y_stress,
        color='lightblue', linestyle=':')
    ax.legend(loc=(1.1, 0))
    return ax


#function to run all values with zero parameters to test pymf6
def do_all(model_path, name, wel_q=0, wel_c=0, mixelm=-1, verbosity_level=0):
    """Do all steps:

    * create model input files
    * run the simulation
    * show calculated heads as map
    * show head at well over time
    """
    wel_coords = (0, 4, 4)
    make_input(
        wel_q=wel_q,
        wel_coords=wel_coords,
        wel_c=wel_c,
        model_path=model_path,
        name=name,
        mixelm=mixelm,
        verbosity_level=verbosity_level
        )
    run_simulation(
        model_path=model_path,
        verbosity_level=verbosity_level
        )
    show_heads(model_path, name)
    show_well_head(model_path, name, wel_coords)

