# .----------------.  .----------------.  .----------------. 
#| .--------------. || .--------------. || .--------------. |
#| |     _____    | || |     ______   | || |  _________   | |
#| |    |_   _|   | || |   .' ___  |  | || | |  _   _  |  | |
#| |      | |     | || |  / .'   \_|  | || | |_/ | | \_|  | |
#| |      | |     | || |  | |         | || |     | |      | |
#| |     _| |_    | || |  \ `.___.'\  | || |    _| |_     | |
#| |    |_____|   | || |   `._____.'  | || |   |_____|    | |
#| |              | || |              | || |              | |
#| '--------------' || '--------------' || '--------------' |
# '----------------'  '----------------'  '----------------' 
#          Temperature Gradient Plotting Script
#                  By: Johann J Cardenas
# '----------------'  '----------------'  '----------------' 

import numpy as np
import matplotlib.pyplot as plt

##################################################
########            User Inputs           ######## 
##################################################

# Global Variables 
aa1 = 0.0021   # ThermaL Conductivity, AC Layer [Kcal/hmC]
aa2 = 0.0030   # ThermaL Conductivity, Base Layer [Kcal/hmC]
ll1 = 1.38     # Thermal Diffusivity, AC Layer [mm^2/s]
ll2 = 1.00     # Thermal Diffusivity, Base Layer [mm^2/s]
H = 0.085      # Thickness of AC Layer [m]
c = 18         # Temperature at the surface [C]  
Tmax = 21.00    # Maximum Temperature at the bottom [C]
Tmin = 15.00
t = 6

# Functions U, FFs, Fs as provided in the user's MATLAB code
def U(z, s, t):
    r1 = np.sqrt(s / aa1)
    r2 = np.sqrt(s / aa2)
    llrp = 1 + ll1 * r1 / ll2 / r2
    llrm = 1 - ll1 * r1 / ll2 / r2
    L = llrp / llrm
    A = (Fs(s, t) - c / s) / (1 - L * np.exp(2 * H * r1))
    return A * (np.exp(r1 * z) - L * np.exp(r1 * (2 * H - z)))

def FFs(p, z, t):
    return p / t * U(z, p / t, t)

def Fs(s, t):
    return (Tmax + Tmin) / 2 / s + 6 * np.pi * (Tmax - Tmin) / (np.pi**2 + 144 * s**2)

# Translating the main MATLAB script
# Defining parameters for the quadrature formula
Pj = np.array([128.3767707781087 + 1j * 16.66062584162301,
               128.3767707781087 - 1j * 16.66062584162301,
               122.2613148416215 + 1j * 50.12719263676864,
               122.2613148416215 - 1j * 50.12719263676864,
               109.3430343060001 + 1j * 84.09672996003092,
               109.3430343060001 - 1j * 84.09672996003092,
               87.76434640082609 + 1j * 119.2185389830121,
               87.76434640082609 - 1j * 119.2185389830121,
               52.25453367344361 + 1j * 157.2952904563926,
               52.25453367344361 - 1j * 157.2952904563926])/10

Wj = np.array([-8684.606112670226 + 1j * 154574.2053305275,
               -8684.606112670226 - 1j * 154574.2053305275,
               15516.34444257753 - 1j * 84398.32902983925,
               15516.34444257753 + 1j * 84398.32902983925,
               -8586.520055271992 + 1j * 23220.65401339348,
               -8586.520055271992 - 1j * 23220.65401339348,
               1863.271916070924 - 1j * 2533.223820180114,
               1863.271916070924 + 1j * 2533.223820180114,
               -103.4901907062327 + 1j * 41.10935881231860,
               -103.4901907062327 - 1j * 41.10935881231860])/10


##################################################
########          Vertical Mesh           ######## 
##################################################

# Layer 1
NElem = 9
Thick = 50
Bias = 1.01

Zn = np.zeros(NElem + 1)
Z = np.zeros(NElem + 1)
Zn[1] = Thick * (Bias ** (1 / (NElem - 1)) - 1) / (Bias ** (NElem / (NElem - 1)) - 1)
Z[1] = Zn[1] / 1000

for i in range(2, NElem + 1):
    Zn[i] = Zn[i - 1] * Bias ** (1 / (NElem - 1))
    Z[i] = np.sum(Zn[:i + 1]) / 1000

# Layer 2
NElem2 = 8
Thick2 = 50
Bias2 = 1.30

Zn = np.pad(Zn, (0, NElem2), 'constant')  # Extend the Zn array for Layer 2
Z = np.pad(Z, (0, NElem2), 'constant')  # Extend the Z array for Layer 2
Zn[NElem + 1] = Thick2 * (Bias2 ** (1 / (NElem2 - 1)) - 1) / (Bias2 ** (NElem2 / (NElem2 - 1)) - 1)
Z[NElem + 1] =Z [NElem] + Zn[NElem + 1]/1000

for i in range(NElem + 2, NElem + NElem2 + 1):
    Zn[i] = Zn[i - 1] * Bias2 ** (1 / (NElem2 - 1))
    Z[i] = np.sum(Zn[:i + 1]) / 1000


# Layer 3



Zn, Z  # Displaying the arrays to confirm successful translation

# Solving the Integral
Temp = np.zeros(len(Z))
for tt in range(len(Z)):
    z = Z[tt]
    Int = 0
    for ii in range(10):
        p = Pj[ii]
        w = Wj[ii]
        Fj = FFs(p, z, t)
        Int1 = w * Fj
        Int += Int1
    Temp[tt] = Int
    Int=0

TOC = np.column_stack((1000 * Z, Temp + c))
TOC

# Printing the Output Table
print("Z[mm]  T[C]")
for r in range(len(Z)):
    print(f"{1000 * Z[r]:.2f}  {Temp[r] + c:.2f}")

# Note: The plotting part will be translated next, using matplotlib in Python
# For now, let's check the output of the translated part
TOC  # Displaying the Temperature-Depth table

# Plotting the results
plt.figure()

plt.plot(Temp + c, Z * 1000, "-o", color='r', markersize=5, 
         markeredgecolor='red', markerfacecolor=[1, 0.6, 0.6], label='LV-P1-SL0')
plt.xlabel('Temperature (C)', fontweight=str('bold'))
plt.ylabel('Depth (mm)', fontweight=str('bold'))
plt.legend(loc='upper right')
plt.xlim([19, 21])
plt.ylim([-10, Thick + Thick2])
plt.gca().invert_yaxis()  # Reversing the Y-axis
plt.grid(True, which='both', linestyle='--', linewidth='0.5')
plt.minorticks_on()


plt.show()
