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

# Given Prony series terms for N90-0%ABR
taus = np.array([1.05E+05, 1.71E+04, 2.78E+03, 4.52E+02, 7.36E+01, 1.20E+01, 1.94E+00,
                 3.16E-01, 5.14E-02, 8.36E-03, 1.36E-03, 2.21E-04, 3.59E-05, 5.84E-06, 9.50E-07])
En = np.array([24.22, 39.67, 85.64, 193.26, 452.33, 1019.08, 2013.43, 3212.43, 4010.34,
               4011.34, 3389.49, 2555.12, 1787.98, 1200.05, 772.68])

# WLF coefficients for N90-0%ABR
C1, C2 = 33.3993, 294.2

# Reference temperature (variable for later adjustment)
T_ref = 21  # Reference temperature in degrees Celsius

def wlf_shift_factors(T, C1, C2, T_ref):
    """
    Calculate the shift factor for a given temperature using the WLF equation.
    """
    aT = 10**(-C1 * (T - T_ref) / (C2 + (T - T_ref)))
    return aT

def relaxation_modulus(taus, En, time):
    """
    Calculate the relaxation modulus using the Prony series.
    """
    E_relax = En[0]
    for tau, E in zip(taus[1:], En[1:]):
        E_relax += E * np.exp(-time / tau)
    return E_relax

# Frequency range for the master curve (reduced frequency)
reduced_frequency = np.logspace(-6, 6, num=20)  # from 1E-05 to 1E+06

# Calculate the shift factor at the reference temperature
aT_ref = wlf_shift_factors(T_ref, C1, C2, T_ref)

# Convert reduced frequency to reduced time (inverse relationship)
reduced_time = 1 / (reduced_frequency * aT_ref)

# Calculate the relaxation modulus at each reduced time
E_relaxation = relaxation_modulus(taus, En, reduced_time)

# Plot the master curve
plt.figure(figsize=(4, 3), dpi=300)
plt.loglog(reduced_frequency, E_relaxation, '-ro',label=f'T_ref={T_ref}°C')
plt.xlabel('Reduced Frequency (Hz)')
plt.ylabel('Complex Modulus (MPa)')
plt.title('Dynamic Modulus Master Curve for N90-0%ABR')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Output a text file, with the values for the reduced frequency and the relaxation modulus
np.savetxt('master_curve.txt', np.column_stack((reduced_frequency, E_relaxation)), header='Reduced Frequency (Hz), Complex Modulus (MPa)', delimiter=',', fmt='%1.8e')
