#############            __     ______     ______            ##############
#############           /\ \   /\  ___\   /\__  _\           ##############
#############           \ \ \  \ \ \____  \/_/\ \/           ##############
#############            \ \_\  \ \_____\    \ \_\           ##############
#############             \/_/   \/_____/     \/_/           ##############                       
   
#####             PRONY SERIES FROM SIGMOIDAL  PARAMETERS            ######
################                                           ################
########               Authors: Johann J Cardenas                ##########
###########################################################################

"""
Last updated on March 14th, 2024.
Last modified by Johann J Cardenas
"""

import numpy as np
import matplotlib.pyplot as plt

# Polynomial coefficients for the shift factor equation
a = -1.34707e-4
b = -1.54514e-1
c = 3.14417

# Function to calculate the shift factor using the polynomial equation
def shift_factor_polynomial(T):
    return a * T**2 + b * T + c

# Convert psi to MPa for plotting
def psi_to_mpa(psi):
    return psi * 0.00689476

# Curve fitting parameters for the Master Curve equation
alpha = 3.929687
beta = -1.30274
gamma = 0.43318
delta = 2.83471

# Function to calculate dynamic modulus using the Master Curve equation
def calculate_dynamic_modulus(frequencies, alpha, beta, gamma, delta, log_aT):
    log_E_star = delta + np.log10(alpha) / (1 + np.exp(-beta - gamma * (np.log10(frequencies) + log_aT)))
    return 10 ** log_E_star

# Original frequencies and dynamic modulus data in psi
original_frequencies = np.array([25, 10, 1, 0.1, 0.01])  # Hz
E_star_psi_20C = np.array([1_899_950, 1_515_370, 858_162, 383_678, 169_287])
E_star_psi_4C = np.array([4_017_028, 3_484_484, 2_719_819, 1_891_369, 1_152_713])
E_star_psi_10C = np.array([5_194_136, 4_667_716, 4_014_271, 3_380_972, 2_833_553])

# Calculate shift factors for 4°C and -10°C
log_aT_4C = shift_factor_polynomial(4)
log_aT_10C = shift_factor_polynomial(-10)

# Calculate reduced frequencies for 4°C and -10°C
reduced_frequencies_4C = original_frequencies * 10**(log_aT_4C)
reduced_frequencies_10C = original_frequencies * 10**(log_aT_10C)

# Calculate the master curve over a range of reduced frequencies
reduced_frequencies_master = np.logspace(-5, 6, num=1000)  # From 1E-05 to 1E+06
E_star_master_curve = calculate_dynamic_modulus(reduced_frequencies_master, alpha, beta, gamma, delta, 0)
E_star_master_curve_mpa = psi_to_mpa(E_star_master_curve)

# Plot the master curve
plt.figure(figsize=(4, 3), dpi=300)

# Plotting the data points for 20°C, 4°C, and -10°C adjusted to the master curve at reference temperature 20°C
plt.loglog(original_frequencies, psi_to_mpa(E_star_psi_20C), 'o', color='red', label='Data at 20°C')
plt.loglog(reduced_frequencies_4C, psi_to_mpa(E_star_psi_4C), 'o', color='green', label='Data at 4°C Shifted to 20°C')
plt.loglog(reduced_frequencies_10C, psi_to_mpa(E_star_psi_10C), 'o', color='blue', label='Data at -10°C Shifted to 20°C')

# Format plot
plt.xlabel('Reduced Frequency (Hz)')
plt.ylabel('Dynamic Modulus (MPa)')
plt.title('Dynamic Modulus Master Curve at 20°C with Shifted Data')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()



E_star_mpa_20C = psi_to_mpa(E_star_psi_20C)
E_star_mpa_4C = psi_to_mpa(E_star_psi_4C)
E_star_mpa_10C = psi_to_mpa(E_star_psi_10C)

# Combine the reduced frequencies and corresponding modulus data into two columns
combined_reduced_frequencies = np.concatenate((original_frequencies, reduced_frequencies_4C, reduced_frequencies_10C))
combined_E_star_mpa = np.concatenate((E_star_mpa_20C, E_star_mpa_4C, E_star_mpa_10C))

# Since the reduced frequencies for the 20°C data are just the original frequencies, we don't need to adjust them
# Now, save this information into a text file
data_to_save = np.column_stack((combined_reduced_frequencies, combined_E_star_mpa))
np.savetxt('DynamicModulusData.txt', data_to_save, header='Reduced Frequency (Hz)\tDynamic Modulus (MPa)')
