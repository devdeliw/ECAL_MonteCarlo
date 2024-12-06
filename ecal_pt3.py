import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.special import gamma

# material and simulation parameters
X0 = 0.89  # radiation length in cm
density = 8.28  # density in g/cm^3
dEdx = 11.5/X0  # ionization energy loss in MeV/cm

# simulation parameters
E_initial = 1000.0  # initial energy in MeV
num_events = 1000  # number of simulated events
calorimeter_depth = 25.0  # depth in cm
num_bins = 250  # number of spatial bins
bin_width = calorimeter_depth / num_bins  # width of each bin in cm

# helper functions
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

def update_particle_position(particle, distance, charged_counts, photon_counts):
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

# simulation function
def simulate_event(E_initial):
    particle_stack = []
    primary_electron = {'type': 'electron', 'energy': E_initial, 'position': 0.0}
    particle_stack.append(primary_electron)

    charged_particle_counts = np.zeros(num_bins)
    photon_counts = np.zeros(num_bins)
    total_energy_deposited = 0.0

    while particle_stack:
        particle = particle_stack.pop()
        while particle['energy'] > 0 and particle['position'] < calorimeter_depth:
            if particle['type'] in ['electron', 'positron']:
                mean_free_path = X0
                distance = get_interaction_length(mean_free_path)
                distance = min(distance, calorimeter_depth - particle['position'])
                ion_loss = ionization_loss(distance)
                particle['energy'] -= ion_loss
                total_energy_deposited += ion_loss/3 # 3 comes from post-calibration
                if particle['energy'] <= 0:
                    break
                update_particle_position(particle, distance, charged_particle_counts, photon_counts)
                photon = bremsstrahlung(particle)
                particle_stack.append(photon)
            elif particle['type'] == 'photon':
                mean_free_path = (9.0/7.0) * X0
                distance = get_interaction_length(mean_free_path)
                distance = min(distance, calorimeter_depth - particle['position'])
                update_particle_position(particle, distance, charged_particle_counts, photon_counts)
                electron, positron = pair_production(particle)
                particle_stack.extend([electron, positron])
                break
            else:
                break

    return charged_particle_counts, total_energy_deposited

# running the simulation
def run_simulation_for_energies(energies_GeV, num_events=1000):
    # Runs the simulation for a list of initial energies in MeV
    results = {}
    energy_deposits = {}
    for E_GeV in energies_GeV:
        E_initial = E_GeV*1000
        print(f"\nRunning simulation for E0 = {E_GeV} GeV...")
        total_charged_counts = np.zeros(num_bins)
        deposits = []
        for event in tqdm(range(num_events), desc=f"Simulating {E_GeV} GeV"):
            charged_counts, total_energy = simulate_event(E_initial)
            total_charged_counts += charged_counts
            deposits.append(total_energy)
        average_charged_counts = total_charged_counts / num_events
        results[E_GeV] = average_charged_counts
        energy_deposits[E_GeV] = deposits
    return results, energy_deposits

def calculate_statistics(energy_deposits):
    # calculates the mean and standard deviation of energy deposits for each incident energy
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

def plot_energy_deposition(means_GeV, sigmas_GeV, energies_GeV):
    # plots the relationship between incident energy and mean energy deposition with error bars.
    plt.figure(figsize=(12, 8))
    plt.errorbar(energies_GeV, means_GeV, yerr=sigmas_GeV, color='purple', 
                 capsize = 5, capthick=1, label='Simulation Data', alpha = 0.5)
    
    # perform linear fit
    coefficients = np.polyfit(energies_GeV, means_GeV, 1)
    poly = np.poly1d(coefficients)
    fit_line = poly(energies_GeV)
    plt.plot(energies_GeV, fit_line, 'r--', label=f'Linear Fit: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
    
    plt.xlabel('Incident Energy $E_0$ (GeV)', fontsize=14)
    plt.ylabel('Mean Energy Deposited (GeV)', fontsize=14)
    plt.title('Energy Deposition vs. Incident Energy in CMS ECAL', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/part2_c.png', dpi=300)
    plt.close()

    fig, axis = plt.subplots(1, 1, figsize = (12, 8))
    plt.plot(energies_GeV, sigmas_GeV, 'o-', color = 'red', label = 'Resolution')


    def resolution_model(E, a, b):
        return a * np.sqrt(E) + b

    # perform the curve fitting
    popt, pcov = curve_fit(resolution_model, energies_GeV, sigmas_GeV)
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    energy_fit = np.linspace(np.min(energies_GeV), np.max(energies_GeV), 100)
    sigma_fit = resolution_model(energy_fit, a_fit, b_fit)
    plt.plot(energy_fit, sigma_fit, '-', color='blue', label=f'Fit: σ = {a_fit:.3f}√E + {b_fit:.3f}')


    plt.xlabel('Incident Energy $E_0$ (GeV)', fontsize=14)
    plt.ylabel('Calorimeter Resolution (GeV)', fontsize=14)
    plt.title('Calorimeter Resolution vs. Incident Energy in CMS ECAL', fontsize=16) 
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('images/part2_d.png', dpi=300)
    plt.close()
    
    # calculate and display the linearity
    slope, intercept = coefficients
    print(f"Linear Fit Equation: y = {slope:.2f}x + {intercept:.2f} GeV")
    print(f"Coefficient of Determination (R²): {calculate_r_squared(energies_GeV, means_GeV, poly):.4f}")

def calculate_r_squared(energies_GeV, means_GeV, poly):
    predicted = poly(energies_GeV)
    ss_res = np.sum((means_GeV - predicted) ** 2)
    ss_tot = np.sum((means_GeV - np.mean(means_GeV)) ** 2)
    R_squared = 1 - (ss_res / ss_tot)
    return R_squared

def plot_energy_resolution(energy_deposits):
    energy_deposits_GeV = np.array(energy_deposits) / 1000.0

    mean_E = np.mean(energy_deposits_GeV)
    sigma_E = np.std(energy_deposits_GeV)

    plt.figure(figsize=(12, 8))
    counts, bins, patches = plt.hist(energy_deposits_GeV, bins=50, density=True, alpha=0.5, 
                                     color='r', label='Energy Deposits')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (sigma_E * np.sqrt(2 * np.pi))) * np.exp(- (x - mean_E)**2 / (2 * sigma_E**2))
    plt.plot(x, p, 'r-', linewidth=2, label='gaussian fit')

    title = f"Energy Resolution: mean = {mean_E:.3f} GeV, σ = {sigma_E:.3f} GeV"
    plt.title(title, fontsize=14)
    plt.xlabel('Total energy deposited in ECAL (GeV)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('images/part2_a.png', dpi=300)
    plt.show()

def plot_average_densities(results):
    # plots the average charged particle densities for multiple energies on the same plot
    # for part b of part 2
    plt.figure(figsize=(12, 8))
    color = iter(cm.rainbow(np.linspace(0, 1, len(results))))
    for E_GeV, densities in results.items():
        depths = np.linspace(0, calorimeter_depth, num_bins)
        c = next(color)
        plt.plot(depths, densities, label=f'E0 = {E_GeV} GeV', color=c, linewidth=2)

    plt.xlabel('Calorimeter Depth (cm)', fontsize=14)
    plt.ylabel('Avg. Number of Charged Particles', fontsize=14)
    plt.title('Longitudinal Development of Electromagnetic Showers in CMS ECAL', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/part2_b.png', dpi=300)
    plt.show()


#--------#
# PART 3 #
#--------#
from scipy.optimize import curve_fit
from scipy.special import gamma

def energy_deposition_function(t, a, b):
    return b * (b * t)**(a - 1) * np.exp(-b * t) / gamma(a) # independent of E_0 because of normalization

def fit_energy_deposition_function(results, initial_energies_GeV, X0):
    total_counts_1GeV = np.sum(results[1])
    fit_parameters = {}
    
    # compute depths 
    depths = (np.arange(num_bins) + 0.5) * bin_width
    t = depths / X0  # t = x / X0

    for E_GeV in initial_energies_GeV:
        counts_per_bin = results[E_GeV]
        # normalize to 1GeV
        counts_norm = counts_per_bin / total_counts_1GeV
        counts_norm = counts_norm / E_GeV
        
        ydata = counts_norm
        xdata = t
        mask = ydata > 0 # avoid log issues
        xdata = xdata[mask]
        ydata = ydata[mask]

        # initial ansatz guess
        initial_guess = [2.0, 1.0]
        try:
            popt, pcov = curve_fit(energy_deposition_function, xdata, ydata, p0=initial_guess)
            a_fit, b_fit = popt
            fit_parameters[E_GeV] = {'a': a_fit, 'b': b_fit}
            print(f"Fitted parameters for E0 = {E_GeV} GeV: a = {a_fit:.4f}, b = {b_fit:.4f}")
        except RuntimeError as e:
            print(f"Fit failed for E0 = {E_GeV} GeV: {e}")
            fit_parameters[E_GeV] = {'a': np.nan, 'b': np.nan}
    return fit_parameters

if __name__ == "__main__":
    initial_energies_GeV = [1, 3, 5, 10]
    num_events_per_energy = 1000

    results, energy_deposits = run_simulation_for_energies(initial_energies_GeV, num_events=num_events_per_energy)
    means_GeV, sigmas_GeV, energies_GeV = calculate_statistics(energy_deposits)
    plot_energy_deposition(means_GeV, sigmas_GeV, energies_GeV)
    
    fit_parameters = fit_energy_deposition_function(results, initial_energies_GeV, X0)
    
    print("\nFitted parameters for energy deposition function:")
    print("{:<10} {:<10} {:<10}".format('E0 (GeV)', 'a', 'b'))
    for E_GeV in initial_energies_GeV:
        a_fit = fit_parameters[E_GeV]['a']
        b_fit = fit_parameters[E_GeV]['b']
        print("{:<10} {:<10.4f} {:<10.4f}".format(E_GeV, a_fit, b_fit))
