"""
Convenience functions for SPA simulations.
"""

import numpy as np
import pandas as pd
from scipy.stats import sem
from tqdm.notebook import tqdm

import centrex_TlF as centrex


def run_traj_ensemble_simulation(df_traj: pd.DataFrame, odepars, obe_system, exc, rho, transition_name,
                                 laser_detunings: np.ndarray = np.linspace(-40,40,101)*2*np.pi*1e6,
                                 n_traj = 10, save = True, pb=False):
    """
    Runs a frequency sweep for n_traj trajectories in df_traj and returns
    number of fluoresced photons for eac frequency and trajectory.
    """

    # Make dataframe that contains trajectory and laser detuning values
    df_traj_ens = df_traj.sample(n = n_traj).reset_index().merge(pd.Series(laser_detunings, name = "laser_detuning").to_frame(), how = 'cross')
    df_traj_ens['detuning'] = df_traj_ens.doppler + df_traj_ens.laser_detuning
    

    # Function that gets the output
    output_func = centrex.lindblad.setup_state_integral_calculation(
                    states = exc[0].get_indices(obe_system.QN, mode = "python"),
                    nphotons = True
                )

    # Callback function that stops the simulation
    zmax = 10*odepars.σzlaser
    cb = centrex.lindblad.setup_discrete_callback_terminate(odepars, f"vz*t >= {zmax}")

    # Set scan parameters
    scan_params = ["δl", "y0", "vz", "vy"]
    scan_values = [
        df_traj_ens.detuning.values,
        df_traj_ens.y.values,
        df_traj_ens.vz.values,
        df_traj_ens.vy.values,
        ]
    
    # Define time range
    tspan = (0, df_traj_ens.delta_t.max())
    
    # Define ensemble problem
    ens_prob = centrex.lindblad.setup_problem_parameter_scan(
                            odepars, tspan, rho, scan_params, scan_values, 
                            dimensions = 1,
                            output_func = output_func,
                            zipped = True)
    
    # Run the simulation
    centrex.lindblad.solve_problem_parameter_scan(ensemble_problem_name = ens_prob, saveat = 1e-7,
                                                    callback=cb)
    
    # Get array with number of photons for each laser detuning and trajectory
    n_photons = centrex.lindblad.get_results_parameter_scan()

    # Add number of photons as a column
    df_traj_ens['n_photons'] = n_photons

    # Save results
    if save:
        time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
        fname = f"./saved_data/{transition_name}_traj_ens_n={n_traj}_{time}.csv"
        with open(fname,'w+', encoding="utf-8") as f:
            f.write(odepars.__repr__()+'\n',)
            df_traj_ens.to_csv(f, index=False)

    return df_traj_ens

def bootstrap_frequency_scan(df_traj: pd.DataFrame, odepars, obe_system, exc, rho, transition_name,
                             laser_detunings: np.ndarray = np.linspace(-40,40,101)*2*np.pi*1e6,
                             n_traj = 10, n_bs = 2, save = True):
    """
    Repeat a frequency scan multiple times with different random sample
    of from df_traj to get bootstrapped error bars.
    """
    df_bs = pd.DataFrame()
    df_raw = pd.DataFrame()

    for n in tqdm(range(n_bs)):
        # Run simulation for ensemble of trajectories
        df_traj_ens = run_traj_ensemble_simulation(df_traj, odepars, obe_system, exc, rho, transition_name,
                                 laser_detunings=laser_detunings,
                                 n_traj = n_traj, save = False, pb = True)

        # Append to dataframe that contains all trajectories and raw data for them
        df_raw = pd.concat([df_raw, df_traj_ens])

        # Calculate mean number of photons for each frequency
        mean_photons = df_traj_ens.groupby('laser_detuning').mean().reset_index()

        # Concatenate to bootstrap result dataframe
        df_bs = pd.concat([df_bs, mean_photons])

    df_bs = df_bs.reset_index()

    # Calculate mean and standard error from bootstrapped results
    df_agg = (df_bs[['laser_detuning', 'n_photons']].groupby('laser_detuning')
              .agg({"n_photons":[np.mean, sem]})).reset_index()

    # Save data
    if save:
        time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
        fname = f"./saved_data/{transition_name}_bs_n_traj={n_traj}_n_bs={n_bs}_{time}.csv"
        with open(fname,'w+', encoding="utf-8") as f:
            f.write(odepars.__repr__()+'\n',)
            df_bs.to_csv(f, index=False)

        fname = f"./saved_data/{transition_name}_bs_agg_n_traj={n_traj}_n_bs={n_bs}_{time}.csv"
        with open(fname,'w+', encoding="utf-8") as f:
            f.write(odepars.__repr__()+'\n',)
            df_agg.to_csv(f, index=False)

        fname = f"./saved_data/{transition_name}_bs_raw_n_traj={n_traj}_n_bs={n_bs}_{time}.csv"
        with open(fname,'w+', encoding="utf-8") as f:
            f.write(odepars.__repr__()+'\n',)
            df_raw.to_csv(f, index=False)

    return df_agg, df_bs, df_raw
    

