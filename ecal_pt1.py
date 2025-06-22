import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.special import gamma 

"""

    Describing Calorimeter Response using a Monte Carlo Model of Electromagnetic
    Showers

    The goal of this project is to simulate (with many assumptions) the
    performance of the CMS electromagnetic calorimeter & predict its resolution
    and linearity.

    Writing a Monte Carlo simulation that predicts the *longitudinal development
    of an EM shower in the CMS ECAL.

    Simplifying Assumptions: 
    ------------------------
    - Calorimeter is uniform crystal of lead tungstate 25 centimeters deep 
    - 1 Dimensional Model ignoring transverse spreading 
    - Discrete Bremsstrahlung with probability inversely propoertional to
      radiation length. 
    - Energy gets divided equally between outgoing electron and photon. 
        - Ionization energy loss per cm is constant. 
    - Pair production is the only process that matters with the probability of
      pair production occuring inversely proportional to 9/7*radiation length. 

"""

plt.rcParams['axes.facecolor'] = '#fafafa'  
plt.rcParams['axes.edgecolor'] = 'gray' 
plt.rcParams['grid.color'] = 'gray'     
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'


X0 = 0.89                   # Radiation length in cm for lead tungstate
density = 8.28              # Density in in g/cm^3 for lead tungstate
dEdx = 11.5/X0              # Ionization energy loss in MeV/cm

E_initial = 1000.0          # Initial energy in MeV (1 GeV)
num_events = 1000           # Number of events to simulate
calorimeter_depth = 25.0    # Calorimeter Depth in cm
num_bins = 250              # Number of spatial bins/"planes" across calorimeter 
bin_width = calorimeter_depth / num_bins 

def get_interaction_length(mean_free_path): 
    # random interaction length based on mean free path 
    return np.random.exponential(mean_free_path) 

def ionization_loss(distance): 
    return dEdx * distance 

def bremsstrahlung(electron): 
    # simulates bremsstrahlung for an electron 
    # splits energy equally between electron and emitted photon 
    energy = electron['energy'] / 2.0 
    position = electron['position'] 

    # generate a photon and update electron energy 
    photon = {'type': 'photon', 'energy': energy, 'position': position} 
    electron['energy'] = energy 

    return photon 

def pair_production(photon): 
    # simulate pair production for a photon
    # positron and electron share the energy of the photon equal 
    energy = photon['energy']/2 
    position = photon['position'] 

    # electron and positron creation 
    electron = {'type': 'electron', 'energy': energy, 'position': position} 
    positron = {'type': 'positron', 'energy': energy, 'position': position}

    return electron, positron 

def update_particle_position(particle, distance, charged_counts, photon_counts):
    # updates particle positions including photon counts 
    initial_bin = int(particle['position'] / bin_width)
    final_position = particle['position'] + distance
    final_bin = int(final_position / bin_width)
    particle['position'] = final_position

    # planes crossed 
    bins_crossed = range(min(initial_bin, final_bin), max(initial_bin, final_bin) + 1)
    for b in bins_crossed:
        if 0 <= b < num_bins:
            if particle['type'] in ['electron', 'positron']:
                charged_counts[b] += 1
            elif particle['type'] == 'photon':
                photon_counts[b] += 1

# simulation 
def simulate_event():
    # initialize particle stack with the primary electron
    particle_stack = []
    primary_electron = {'type': 'electron', 'energy': E_initial, 'position': 0.0}
    particle_stack.append(primary_electron)

    # initialize arrays to record counts
    charged_particle_counts = np.zeros(num_bins)
    photon_counts = np.zeros(num_bins)

    while particle_stack:
        particle = particle_stack.pop()
        while particle['energy'] > 0 and particle['position'] < calorimeter_depth:
            if particle['type'] in ['electron', 'positron']:
                # mean free path for bremsstrahlung (i.e. radiation length)
                mean_free_path = X0
                # distance to next interaction
                distance = get_interaction_length(mean_free_path)
                # ensure particle doesn't exceed calorimeter
                distance = min(distance, calorimeter_depth - particle['position'])
                # energy loss due to ionization
                ion_loss = ionization_loss(distance)
                particle['energy'] -= ion_loss
                
                if particle['energy'] <= 0:
                    break  # particle stops, gets aborbed 
                # update position and record crossings
                update_particle_position(particle, distance, charged_particle_counts, photon_counts)

                # perform bremsstrahlung
                interaction_prob = distance / X0 
                if np.random.uniform(0, 1) < interaction_prob: 
                    photon = bremsstrahlung(particle)
                    particle_stack.append(photon)
                    
            elif particle['type'] == 'photon':
                # mean free path for pair production
                mean_free_path = (9.0 / 7.0) * X0
                # distance to next interaction
                distance = get_interaction_length(mean_free_path)
                # ensure photon doesn't exceed calorimeter
                distance = min(distance, calorimeter_depth - particle['position'])
                # update position and record crossings
                update_particle_position(particle, distance, charged_particle_counts, photon_counts)

                # perform pair production
                interaction_prob = distance / mean_free_path 
                if np.random.uniform(0, 1) < interaction_prob: 
                    electron, positron = pair_production(particle)
                    particle_stack.extend([electron, positron])
                break  # photon is converted into pair; stop processing this photon
            else:
                break  

    return charged_particle_counts, photon_counts

def run_simulation():
    total_charged_counts = np.zeros(num_bins)
    total_photon_counts = np.zeros(num_bins)
    for event in range(num_events):
        charged_counts, photon_counts = simulate_event()
        total_charged_counts += charged_counts
        total_photon_counts += photon_counts
        if (event + 1) % 100 == 0:
            print(f"Completed {event + 1} / {num_events} events")

    # calculate averages
    average_charged_counts = total_charged_counts / num_events
    average_photon_counts = total_photon_counts / num_events

    return average_charged_counts, average_photon_counts

def plot_results(average_charged_counts, average_photon_counts):
    depths = np.linspace(0, calorimeter_depth, num_bins)
    plt.figure(figsize=(10, 8))
    plt.plot(depths, average_charged_counts, 'o-', label='Charged Particles (e±)', color='blue', linewidth=2)
    #plt.plot(depths, average_photon_counts, '-', label='Photons', color='red', linewidth=2)

    plt.xlabel('Calorimeter Depth (cm)', fontsize = 14)
    plt.ylabel('Avg. Number of Charged Particles', fontsize = 14)
    plt.title('Simulated Longitudinal Development of Electromagnetic Shower', fontsize = 16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('images/part1_a.png', dpi=300)
    plt.show()



average_charged_counts, average_photon_counts = run_simulation()
plot_results(average_charged_counts, average_photon_counts)



