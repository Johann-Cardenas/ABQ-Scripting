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
import math

##################################################
########            User Inputs           ######## 
##################################################

# Global Variables 
aa1 = 0.0021   # ThermaL Conductivity, AC Layer [Kcal/hmC]
aa2 = 0.0030   # ThermaL Conductivity, Base Layer [Kcal/hmC]
ll1 = 1.38     # Thermal Diffusivity, AC Layer [mm^2/s]
ll2 = 1.00     # Thermal Diffusivity, Base Layer [mm^2/s
c = 18         # Temperature at the surface [C]  
Tmax = 21.00   # Maximum Temperature at the bottom [C]
Tmin = 15.00   # Minimum Temperature at the bottom [C]
t = 6         # Shape Parameter

# AC Structure
Labels = ['SL', 'IML', 'BL']
MyThicks = [40, 75, 40]    # Thickness of each layer [mm]
NElem = [8, 12, 5]         # Number of elements in each layer
Bias = [1.10, 1.30, 1.01]  # Biasing factor for each layer

H = sum(MyThicks)/1000      

##################################################
########    Functions and Parameters      ######## 
##################################################

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

# Parameters of the the quadrature formula
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
########            Calculations          ######## 
##################################################

# Layer 1
Zn = np.zeros(NElem[0] + 1)
Z = np.zeros(NElem[0] + 1)
Zn[1] = MyThicks[0] * (Bias[0] ** (1 / (NElem[0] - 1)) - 1) / (Bias[0] ** (NElem[0]/ (NElem[0] - 1)) - 1)
Z[1] = Zn[1] / 1000

for i in range(2, NElem[0] + 1):
    Zn[i] = Zn[i - 1] * Bias[0] ** (1 / (NElem[0] - 1))
    Z[i] = np.sum(Zn[:i + 1]) / 1000

if len(MyThicks) > 1:
    # Layer 2
    Zn = np.pad(Zn, (0, NElem[1]), 'constant')  # Extend the Zn array for Layer 2
    Z = np.pad(Z, (0, NElem[1]), 'constant')  # Extend the Z array for Layer 2
    Zn[NElem[0] + 1] = MyThicks[1] * (Bias[1] ** (1 / (NElem[1] - 1)) - 1) / (Bias[1] ** (NElem[1] / (NElem[1] - 1)) - 1)
    Z[NElem[0] + 1] =Z [NElem[0]] + Zn[NElem[0] + 1]/1000

    for i in range(NElem[0] + 2, NElem[0] + NElem[1] + 1):
        Zn[i] = Zn[i - 1] * Bias[1] ** (1 / (NElem[1] - 1))
        Z[i] = np.sum(Zn[:i + 1]) / 1000

if len(MyThicks) > 2:
    # Layer 3
    Zn = np.pad(Zn, (0, NElem[2]), 'constant')  # Extend the Zn array for Layer 3
    Z = np.pad(Z, (0, NElem[2]), 'constant')  # Extend the Z array for Layer 3
    Zn[NElem[0] + NElem[1] + 1] = MyThicks[2] * (Bias[2] ** (1 / (NElem[2] - 1)) - 1) / (Bias[2] ** (NElem[2] / (NElem[2]- 1)) - 1)
    Z[NElem[0] + NElem[1] + 1] =Z [NElem[0] + NElem[1]] + Zn[NElem[0] + NElem[1] + 1]/1000

    for i in range(NElem[0] + NElem[1] + 2, NElem[0] + NElem[1] + NElem[2] + 1):
        Zn[i] = Zn[i - 1] * Bias[2] ** (1 / (NElem[2] - 1))
        Z[i] = np.sum(Zn[:i + 1]) / 1000

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

# Put the T values in arrays
Zcoord = np.zeros(len(Z))
for i in range(len(Z)):
    Zcoord[i] = round(Z[i]*1000,2)

ytemp = np.zeros(len(Z))
for i in range(len(Z)):
    ytemp[i] = round(Temp[i] + c,2)
    
print(Zcoord)
print(ytemp)

##################################################
########              PLOT               ######## 
##################################################
# NOTE! Customize the ticks of X and Y axis according to your needs

# Setting up for contour plot
Temp_grid, Depth_grid = np.meshgrid(np.linspace(10, 30, 500), np.linspace(0, sum(MyThicks), 500))
Contour_values = np.zeros_like(Temp_grid)

for i in range(len(Zcoord)):
    depth = Zcoord[i]
    temperature = ytemp[i]
    Contour_values[np.where(Depth_grid >= depth)] = temperature

# Plotting the results
plt.figure(figsize=(3,6))

cbar_ticks = np.arange(math.floor(min(ytemp)), math.ceil(max(ytemp))+0.01, 1.00)  # Values from 18 to 21 with a step of 0.25
levels = np.unique(np.concatenate([cbar_ticks, Contour_values.ravel()]))

contourf_plot = plt.contourf(Temp_grid, Depth_grid, Contour_values, levels=levels, alpha=0.5, cmap='hot_r')

cbar = plt.colorbar(contourf_plot, ticks=cbar_ticks)
cbar.set_label('T (°C)', fontweight=str('bold'))

plt.plot(ytemp, Zcoord, "-o", color='k', markersize=5, 
         markeredgecolor='red', markerfacecolor=[1, 0.6, 0.6])

for i in range(len(MyThicks)):
    plt.axhline(y=sum(MyThicks[:i + 1]), color='k', linestyle='-', linewidth='0.5')
    plt.text(math.floor(min(ytemp))+0.15, sum(MyThicks[:i + 1])-MyThicks[i]+10, Labels[i], bbox=dict(facecolor='white', edgecolor='black', pad=2.0))

plt.ylabel('Depth (mm)', fontweight=str('bold'))
plt.xlabel('Temperature (°C)', fontweight=str('bold'))
plt.xlim([math.floor(min(ytemp)), math.ceil(max(ytemp))])
plt.ylim([0, sum(MyThicks)])

# Add ticks in X and Y
plt.xticks([18, 19, 20, 21])
plt.yticks([0, 25, 50, 75, 100, 125, 150])
plt.gca().invert_yaxis()  # Reversing the Y-axis

plt.tight_layout
plt.savefig('T-Gradient.png', dpi=300, bbox_inches='tight')

plt.show()