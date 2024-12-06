import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.special import gamma

# Set global parameters for axes and figure backgrounds
plt.rcParams['axes.facecolor'] = '#fafafa'  # Light gray background inside axes
plt.rcParams['axes.edgecolor'] = 'gray'  # Gray edges around the axes
plt.rcParams['grid.color'] = 'gray'      # Slightly darker grid lines
plt.rcParams['grid.linestyle'] = '-'    # Solid grid lines
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'


# Material and simulation parameters
X0 = 0.89  # Radiation length in cm
density = 8.28  # Density in g/cm^3
dEdx = 11.5 / X0  # Ionization energy loss in MeV/cm

# Simulation parameters
E_initial = 1000.0  # Initial energy in MeV
num_events = 1000  # Number of simulated events
# We will vary calorimeter_depth in the code below
# num_bins will also vary accordingly
bin_width = 0.1 * X0  # Bin width in cm (fixed fraction of X0)

# Helper functions
def get_interaction_length(mean_free_path):
    return np.random.exponential(mean_free_path)

def ionization_loss(distance):
    loss = dEdx * distance
    return loss

def bremsstrahlung(particle):
    energy = particle['energy'] / 2.0
    position = particle['position']
    photon = {'type': 'photon', 'energy': energy, 'position': position}
    particle['energy'] = energy
    return photon

def pair_production(photon):
    energy = photon['energy'] / 2.0
    position = photon['position']
    electron = {'type': 'electron', 'energy': energy, 'position': position}
    positron = {'type': 'positron', 'energy': energy, 'position': position}
    return electron, positron

def update_particle_position(particle, distance, charged_counts, photon_counts, num_bins, bin_width):
    initial_bin = int(particle['position'] / bin_width)
    final_position = particle['position'] + distance
    final_bin = int(final_position / bin_width)
    particle['position'] = final_position

    bins_crossed = range(min(initial_bin, final_bin), max(initial_bin, final_bin) + 1)
    for b in bins_crossed:
        if 0 <= b < num_bins:
            if particle['type'] in ['electron', 'positron']:
                charged_counts[b] += 1
            elif particle['type'] == 'photon':
                photon_counts[b] += 1

# Simulation function
def simulate_event(E_initial, calorimeter_depth_cm, num_bins, bin_width):
    particle_stack = []
    primary_electron = {'type': 'electron', 'energy': E_initial, 'position': 0.0}
    particle_stack.append(primary_electron)

    charged_particle_counts = np.zeros(num_bins)
    photon_counts = np.zeros(num_bins)
    total_energy_deposited = 0.0

    while particle_stack:
        particle = particle_stack.pop()
        while particle['energy'] > 0 and particle['position'] < calorimeter_depth_cm:
            if particle['type'] in ['electron', 'positron']:
                mean_free_path = X0
                distance = get_interaction_length(mean_free_path)
                distance = min(distance, calorimeter_depth_cm - particle['position'])
                ion_loss = ionization_loss(distance)
                particle['energy'] -= ion_loss
                total_energy_deposited += ion_loss / 3  # 3 comes from post-calibration
                if particle['energy'] <= 0:
                    break
                update_particle_position(particle, distance, charged_particle_counts, photon_counts, num_bins, bin_width)
                photon = bremsstrahlung(particle)
                particle_stack.append(photon)
            elif particle['type'] == 'photon':
                mean_free_path = (9.0 / 7.0) * X0
                distance = get_interaction_length(mean_free_path)
                distance = min(distance, calorimeter_depth_cm - particle['position'])
                update_particle_position(particle, distance, charged_particle_counts, photon_counts, num_bins, bin_width)
                electron, positron = pair_production(particle)
                particle_stack.extend([electron, positron])
                break
            else:
                break

    return charged_particle_counts, total_energy_deposited

# Running the simulation
def run_simulation_for_energies(energies_GeV, num_events, calorimeter_depth_cm, num_bins, bin_width):
    # Runs the simulation for a list of initial energies in MeV
    results = {}
    energy_deposits = {}
    for E_GeV in energies_GeV:
        E_initial = E_GeV * 1000
        print(f"\nRunning simulation for E0 = {E_GeV} GeV...")
        total_charged_counts = np.zeros(num_bins)
        deposits = []
        for event in tqdm(range(num_events), desc=f"Simulating {E_GeV} GeV"):
            charged_counts, total_energy = simulate_event(E_initial, calorimeter_depth_cm, num_bins, bin_width)
            total_charged_counts += charged_counts
            deposits.append(total_energy)
        average_charged_counts = total_charged_counts / num_events
        results[E_GeV] = average_charged_counts
        energy_deposits[E_GeV] = deposits
    return results, energy_deposits

def calculate_statistics(energy_deposits):
    # Calculates the mean and standard deviation of energy deposits for each incident energy
    energies_GeV = sorted(energy_deposits.keys())
    means_GeV = []
    sigmas_GeV = []
    for E_GeV in energies_GeV:
        deposits_MeV = np.array(energy_deposits[E_GeV])
        mean_MeV = np.mean(deposits_MeV)
        sigma_MeV = np.std(deposits_MeV)
        means_GeV.append(mean_MeV / 1000.0)   # Convert to GeV
        sigmas_GeV.append(sigma_MeV / 1000.0) # Convert to GeV
    return means_GeV, sigmas_GeV, energies_GeV

# Plotting functions
def plot_linearity_vs_depth(all_means, energies_GeV, depth_X0_list):
    plt.figure(figsize=(10, 8))
    for depth_X0 in depth_X0_list:
        means_GeV = all_means[depth_X0]
        plt.plot(energies_GeV, means_GeV, 'o-', label=f'Depth = {depth_X0} X0')
    plt.xlabel('Incident Energy $E_0$ (GeV)', fontsize=14)
    plt.ylabel('Mean Energy Deposited (GeV)', fontsize=14)
    plt.title('Linearity: Energy Deposition vs. Incident Energy for Different Calorimeter Depths', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/linearity_vs_depth.png', dpi=300)

def plot_linearity_ratio_vs_depth(all_means, energies_GeV, depth_X0_list):
    plt.figure(figsize=(10, 8))
    for depth_X0 in depth_X0_list:
        means_GeV = all_means[depth_X0]
        ratios = np.array(means_GeV) / energies_GeV
        plt.plot(energies_GeV, ratios, 'o-', label=f'Depth = {depth_X0} X0')
    plt.xlabel('Incident Energy $E_0$ (GeV)', fontsize=14)
    plt.ylabel('Mean Energy Deposited / Incident Energy', fontsize=14)
    plt.title('Linearity Ratio vs. Incident Energy for Different Calorimeter Depths', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/linearity_ratio_vs_depth.png', dpi=300)

def plot_resolution_vs_depth(all_sigmas, energies_GeV, depth_X0_list):
    plt.figure(figsize=(10, 8))
    for depth_X0 in depth_X0_list:
        sigmas_GeV = all_sigmas[depth_X0]
        plt.plot(energies_GeV, sigmas_GeV, 'o-', label=f'Depth = {depth_X0} X0')
    plt.xlabel('Incident Energy $E_0$ (GeV)', fontsize=14)
    plt.ylabel('Energy Resolution σ (GeV)', fontsize=14)
    plt.title('Resolution vs. Incident Energy for Different Calorimeter Depths', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/resolution_vs_depth.png', dpi=300)

def plot_relative_resolution_vs_depth(all_sigmas, all_means, energies_GeV, depth_X0_list):
    plt.figure(figsize=(10, 8))
    for depth_X0 in depth_X0_list:
        sigmas_GeV = np.array(all_sigmas[depth_X0])
        means_GeV = np.array(all_means[depth_X0])
        relative_resolution = sigmas_GeV / means_GeV
        plt.plot(energies_GeV, relative_resolution, 'o-', label=f'Depth = {depth_X0} X0')
    plt.xlabel('Incident Energy $E_0$ (GeV)', fontsize=14)
    plt.ylabel('Relative Energy Resolution σ / E', fontsize=14)
    plt.title('Relative Energy Resolution vs. Incident Energy for Different Calorimeter Depths', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/relative_resolution_vs_depth.png', dpi=300)

# Main execution
if __name__ == "__main__":
    # Define the list of initial energies in GeV
    initial_energies_GeV = [1, 3, 5, 10]

    # Number of events per energy
    num_events_per_energy = 1000

    # Define the list of calorimeter depths in units of X0
    depth_X0_list = [5, 10, 15, 20, 25]

    # Dictionaries to store results
    all_means = {}
    all_sigmas = {}

    for depth_X0 in depth_X0_list:
        calorimeter_depth_cm = depth_X0 * X0
        num_bins = int(calorimeter_depth_cm / bin_width)
        print(f"\nRunning simulations for calorimeter depth = {depth_X0} X0 ({calorimeter_depth_cm:.2f} cm)")

        # Run the simulation for all energies at this depth
        results, energy_deposits = run_simulation_for_energies(
            initial_energies_GeV,
            num_events_per_energy,
            calorimeter_depth_cm,
            num_bins,
            bin_width
        )

        # Calculate statistics
        means_GeV, sigmas_GeV, energies_GeV = calculate_statistics(energy_deposits)

        # Store the results
        all_means[depth_X0] = means_GeV
        all_sigmas[depth_X0] = sigmas_GeV

    # Plotting the results
    plot_linearity_vs_depth(all_means, energies_GeV, depth_X0_list)
    plot_linearity_ratio_vs_depth(all_means, energies_GeV, depth_X0_list)
    plot_resolution_vs_depth(all_sigmas, energies_GeV, depth_X0_list)
    plot_relative_resolution_vs_depth(all_sigmas, all_means, energies_GeV, depth_X0_list)

