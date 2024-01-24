#############            __     ______     ______            ##############
#############           /\ \   /\  ___\   /\__  _\           ##############
#############           \ \ \  \ \ \____  \/_/\ \/           ##############
#############            \ \_\  \ \_____\    \ \_\           ##############
#############             \/_/   \/_____/     \/_/           ##############                       
   
#######       OBTAIN PRONY SERIES FROM DYNAMIC MODULUS TEST        ########
################             ELECTRIC TRUCKS               ################
########            Prepared by: Johann J Cardenas               ##########
###########################################################################

"""
Last updated on December 17th, 2023.
"""

import numpy as np
import pandas as pd
import math
from lmfit.models import QuadraticModel
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import os.path
from os import path
from scipy.optimize import nnls
import time

plt.close('all') # closes all figures

# Start run time
start = time.time()

#%% Input Parameters
case = 'FD_Mix 1'
row_ct = [6, 6, 6, 6, 6]              # number of frequencies tested per test temp
Test_tempF = [14.0, 39.2, 69.8, 98.6, 129.2] # degF   [-10.0, 4.0, 21.0, 37.0, 54.0] degC

shift_factors = [42.5, 22.5, 0, 18.0, 32.50] # Guesstimate of Shift factors; 0 for reference temp [values should be positive]
rr = np.logspace(-5,8,16)     # seed values for Prony series fitting; initial assumptions are important; check final plot
tref = Test_tempF[1]
To = (tref-32.)*5./9.           # Reference temp, degC
rheo_factor = 2*np.pi           # 1  vs 2*np.pi
Einf = 1.0                      # Long-term modulus, MPa

Test_temp_all = np.concatenate((np.ones(row_ct[0])*Test_tempF[0],
                                np.ones(row_ct[1])*Test_tempF[1],
                                np.ones(row_ct[1])*Test_tempF[2],
                                np.ones(row_ct[1])*Test_tempF[3],
                                np.ones(row_ct[2])*Test_tempF[4]))
# Unit conversion
psi_mpa = 6.8947572932 #  6.89 MPa = 1 ksi
tempC= [(i-32.)*5./9. for i in Test_tempF] ## degC
Test_tempC = [(i-32.)*5./9. for i in Test_temp_all] ## degC

## Units: Test_Temp (degF), Estar (ksi), Delta (degrees)

data = {'Test_TempF': Test_temp_all,
        'Test_TempC': Test_tempC,
        'Frequency':[25.0,  10.0,  5.0,  1.0,  0.5,  0.1,    # test temp = 14.0 degF [-10.0 degC]
                     25.0,  10.0,  5.0,  1.0,  0.5,  0.1,    # test temp = 39.2 degF [4.0 degC]
                     25.0,  10.0,  5.0,  1.0,  0.5,  0.1,    # test temp = 69.8 degF [21.0 degC]
                     25.0,  10.0,  5.0,  1.0,  0.5,  0.1,    # test temp = 98.6 degF [37.0 degC]
                     25.0,  10.0,  5.0,  1.0,  0.5,  0.1],   # test temp = 129.2 degF [54.0 degC]
        'Estar1':[27472.,  26357.,  25495.,  23320.,  22295.,  19864.,    # MPa, C1
                  20444.,  18781.,  17552.,  14646.,  13403.,  10557.,   
                  11920.,  10186.,  9008.,  6469.,  5538.,  3612.,  
                  4833.,  3640.,  2935.,  1687.,   1318.,  723.5,  
                  1911.,  1150.,  838.,  435.1,  341.3,  213.9],   
        'Estar2':[26649.,  25479.,  24576.,  22340.,  21356.,  18911.,    # MPa, C1
                  19781.,  18173.,  16976.,  14194.,  12988.,  10249.,   
                  11051.,  9357.,  8218.,  5825.,   4973.,  3196.,  
                  5281.,  3945.,  3177.,  1770.,   1376.,  731.4,  
                  1932.,  1262.,  937.3,  487.3,  382.8,  234.6],   
        'Estar3':[26582.,  25396.,  24511.,  22350.,  21373.,  19007.,    # MPa, C1
                  19829.,  18400.,  17264.,  14593.,  13426.,  10685.,   
                  10668.,  9056.,  7976.,  5638.,   4819.,  3075.,  
                  4998.,  3729.,  2989.,  1697.,   1314.,  698.7,  
                  1676.,  1003.,  727.4,  387.4,  311.4,  203.0],   
        'Delta1':[4.20,  4.77,  5.14,  6.11,  6.49,  7.78,     # degrees
                  8.08,  9.10,  9.88,  11.81,  12.74,  15.52,
                  15.57,  17.38,  18.50,  21.41,  22.39,  25.17,
                  26.11,  27.76,  28.18,  28.83,  28.33,  27.47,
                  27.66,  30.45,  30.38,  28.13,  26.83,  22.87],
        'Delta2':[4.43,  4.91,  5.23,  6.22,  6.65,  7.98,     # degrees
                  8.22,  9.22,  9.93,  11.95,  12.89,  15.75,
                  16.22,  18.26,  19.52,  22.83,  23.86,  26.90,
                  24.73,  26.91,  27.68,  29.06,  28.77,  28.53,
                  30.23,  31.48,  30.82,  28.59,  27.11,  24.16],
        'Delta3':[4.33,  4.77,  5.11,  6.00,  6.42,  7.62,      # degrees
                  7.64,  8.46,  9.21,  11.19,  12.17,  14.97,
                  16.53,  18.63,  19.95,  23.34,  24.36,  27.18,  
                  25.37,  27.53,  28.20,  29.05,  28.77,  28.25, 
                  28.80,  31.27,  31.05,  28.23,  26.46,  23.33]}


# Create dataframe
Summary_Table = pd.DataFrame(data)

#Estar_avg = psi_mpa * (Summary_Table['Estar1'] + Summary_Table['Estar2']) / 2.0 ## MPa
Estar_avg = (Summary_Table['Estar1'] + Summary_Table['Estar2'] + Summary_Table['Estar3']) / 3.0 ## MPa

Delta_avg = (Summary_Table['Delta1'] + Summary_Table['Delta2'] + Summary_Table['Delta3']) / 3.0 ## degrees

Summary_Table['Dynamic_Modulus'] = Estar_avg
Summary_Table['Phase_Angle'] = Delta_avg


#%% Calculate loss and storage moduli
Summary_Table['E\''] = Summary_Table['Dynamic_Modulus']*np.cos((Summary_Table['Phase_Angle'])*math.pi/180) # Storage modulus
Summary_Table['E\"'] = Summary_Table['Dynamic_Modulus']*np.sin((Summary_Table['Phase_Angle'])*math.pi/180) # Loss modulus

ref_idx = shift_factors.index(0)  # Index of Reference Temperature

# Assign positive or negative sign if before or after reference temp, respectively -- change in shift direction
sf_sign = [-1.0*i if shift_factors.index(i) > ref_idx else i for i in shift_factors]
log_at_val = np.array(sf_sign)/10

log_at = []
for i in range(len(log_at_val)):
    log_at.append(np.ones(row_ct[i])*log_at_val[i])


Summary_Table['Log(at)'] = list(np.concatenate(log_at).flat)
Summary_Table['Log_reduced_freq'] = np.log10(Summary_Table['Frequency']) + Summary_Table['Log(at)']
Summary_Table['Reduced_Freq'] = 10**Summary_Table['Log_reduced_freq']

# locally fit dataset per temperature to find overlap to improve shift factors
def round_nearest(x, a):
    return round(x / a) * a

def fitting_model (x, y, xfit):
    model = QuadraticModel(prefix='bkg_')
    params = model.make_params(a=0, b=0, c=0)
    result = model.fit(np.log(y), params, x=x)
    yfit = np.exp(result.eval(x=xfit))
    return yfit

# Plot local fit per dataset or temp condition for visual check
fig = plt.figure(figsize=(5, 4))

i = 0
x = Summary_Table['Log_reduced_freq'][0:row_ct[i]]
y = Summary_Table['Dynamic_Modulus'][0:row_ct[i]]
xfit = np.arange(3.5, 6.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "b^", label="Test Data" + " " + str(round(tempC[0],1))+" °C")
plt.plot(xfit, yfit, "b--", label="Fitted Model" + " " + str(round(tempC[0],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')
# plt.show()

i = 1
x = Summary_Table['Log_reduced_freq'][row_ct[0]:sum(row_ct[0:i+1])]
y = Summary_Table['Dynamic_Modulus'][row_ct[0]:sum(row_ct[0:i+1])]
xfit = np.arange(1.0, 4.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "gs", label="Test Data" + " " + str(round(tempC[1],1))+" °C")
plt.plot(xfit, yfit, "g--", label="Fitted Model" + " " + str(round(tempC[1],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')
# plt.show()

i = 2
x = Summary_Table['Log_reduced_freq'][sum(row_ct[0:i]):sum(row_ct[0:i+1])]
y = Summary_Table['Dynamic_Modulus'][sum(row_ct[0:i]):sum(row_ct[0:i+1])]
xfit = np.arange(-1.0, 2.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "ro", label="Test Data" + " " + str(round(tempC[2],1))+" °C")
plt.plot(xfit, yfit, "r--", label="Fitted Model" + " " + str(round(tempC[2],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')

i = 3
x = Summary_Table['Log_reduced_freq'][sum(row_ct[0:i]):sum(row_ct[0:i+1])]
y = Summary_Table['Dynamic_Modulus'][sum(row_ct[0:i]):sum(row_ct[0:i+1])]
xfit = np.arange(-3.0, 0.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "kv", label="Test Data" + " " + str(round(tempC[3],1))+" °C")
plt.plot(xfit, yfit, "k--", label="Fitted Model" + " " + str(round(tempC[3],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')

i = 4
x = Summary_Table['Log_reduced_freq'][sum(row_ct[0:i]):sum(row_ct[0:i+1])]
y = Summary_Table['Dynamic_Modulus'][sum(row_ct[0:i]):sum(row_ct[0:i+1])]
xfit = np.arange(-5.0, -1.5, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "m*", label="Test Data" + " " + str(round(tempC[4],1))+" °C")
plt.plot(xfit, yfit, "m--", label="Fitted Model" + " " + str(round(tempC[4],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')

# plt.show()

plt.grid(color = 'lightgrey', linestyle = ':', linewidth = 0.50)
plt.ylabel('Dynamic Modulus (MPa)', fontsize=11.0, fontweight=str('bold'))
plt.xlabel('Reduced Frequency (Hz)', fontsize=11.0, fontweight=str('bold'))

plt.tight_layout()
plt.show()
fig.savefig('ShiftFit_'+case+'.png',dpi=300)
# plt.close()


#%% Write summary table into a text file
txt_file = 'DynamicModSummary_'+case

# check if file exits; if so, clear to overwrite
if os.path.isfile('./'+txt_file+'.txt'):
    f = open(txt_file+'.txt', 'r+')
    f.truncate(0)

# Save Summary_Table pandas dataframe into text file
Summary_Table.to_csv('DynamicModSummary_'+case+'.txt', header=True, index=False, sep='\t', mode='a')

#%% 
# _________________________________________________
# Fitting to Prony Series

Summary_TableT = Summary_Table.T # transposed

tt = Summary_Table['Reduced_Freq']
ww = rheo_factor * tt
pa = Summary_Table['Phase_Angle'] # degrees
Estar = Summary_Table['Dynamic_Modulus'] # MPa; measured data
Ep = Summary_Table['E\'']
Epp = Summary_Table['E\"']

E_0 = max(Estar) ## Instantaneous Modulus

## Referencing Jaime Hernandez' Dynamic Modulus Code
rows, cols = (len(ww),len(rr))

A1 = np.array([[wwi**2 * rrj**2 / (wwi**2 * rrj**2 + 1.0) for rrj in rr ] for wwi in ww])
b1 = Ep - Einf

A2 = np.array([[wwi * rrj / (wwi**2 * rrj**2 + 1.0) for rrj in rr ] for wwi in ww])
b2 = Epp

# Combine matrices and vectors of Storage and Loss Moduli for non-negative least squares
A = np.concatenate((A1, A2), axis=0)
b = np.concatenate((b1, b2), axis=0)

# Solve for Prony Series Terms using on-negative least squares solver from scipy.optimize
PT=nnls(A,b)

# Predict values using estimated Prony Series
www = np.logspace(-5,8,200) #
ttt = 1/(rheo_factor*www) ## convert frequency to time

# Predicted Storage Modulus, E'
Ep_predict = np.array([[wwi**2 * rr[j]**2 * PT[0][j] / (wwi**2 * rr[j]**2 + 1.0) for j in range(len(rr))] for wwi in www])

# Predicted Loss Modulus, E"
Epp_predict = np.array([[wwi * rr[j] * PT[0][j] / (wwi**2 * rr[j]**2 + 1.0) for j in range(len(rr))] for wwi in www])

# Predicted Relaxation Modulus
Erel = np.array([[PT[0][j] * np.exp(-ttt_i / rr[j]) for j in range(len(rr))] for ttt_i in ttt])

Ep_predict, Epp_predict, Erel = np.sum(Ep_predict, 1) + Einf, np.sum(Epp_predict, 1), np.sum(Erel, 1) + Einf
Estar_predict = np.sqrt(Ep_predict**2 + Epp_predict**2)
Ep_pred1, Epp_pred1 = np.sum(A1 * PT[0], axis=1) + Einf, np.sum(A2 * PT[0], axis=1)
Estar_pred1 = np.sqrt(Ep_pred1**2 + Epp_pred1**2)

ERROR_Stor, ERROR_los, ERROR_dyn = np.sum((np.log10(Ep)-np.log10(Ep_pred1))**2),\
                                    np.sum((np.log10(Epp)-np.log10(Epp_pred1))**2),\
                                    np.sum((np.log10(Estar)-np.log10(Estar_pred1))**2)

# Calculate WLF coefficients
tempavgall = np.array([(i-32.)*5./9. for i in Test_tempF])
AA = np.transpose(np.array([tempavgall - To, log_at_val]))
bb = -(tempavgall - To) * log_at_val;
C = nnls(AA, bb)
C = C[0] ## wLF coefficients

T_cal = np.arange(-60, 220, 2)
LogaT_cal = -C[0] * (T_cal - To) / (C[1] + T_cal - To)

# Normalize Prony Series for Abaqus
E_inst = Einf + np.sum(PT[0]) # Instantaneous Modulus, MPa --> summation of long-term plus all Prony coefficients
PT_Abaqus = np.array(PT[0]/E_inst) # Normalize to instantaneous modulus, requirement for Abaqus

#%% Report Fitting Coefficients and Error (SSE)
# Check that the sum of the normalized Prony coefficient sum up to 1
if sum(PT_Abaqus) < 1:
    print('Sum of Normalized Prony Series =', round(sum(PT_Abaqus),5))
else:
    print('Check! Sum of Normalized Prony Series is GREATER THAN ONE! ', round(sum(PT_Abaqus),5))
    
# Summary of Prony Series and WLF Coefficients
d = {'Tau': rr, 'Prony': PT[0], 'Prony_Abaqus': PT_Abaqus}
df = pd.DataFrame(data=d)
Prony_Terms = df[df['Prony'] != 0]

print(Prony_Terms)
print('C1 =', round(C[0],4), '; C2 =', round(C[1],4))
print('Sum of Squared Errors: Estorage =', round(ERROR_Stor, 4), ', Eloss =', round(ERROR_los, 4), ', Edynamic', round(ERROR_dyn, 4))

#%% Plot shifted data, following Summary_Table dataframe

# fine-tuning properties of plot
rows = 2
cols = 3
figsize = (10, 6)
fig, ax = plt.subplots(rows, cols, figsize=figsize)
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()

left  = 0.07   # the left side of the subplots of the figure
right = 0.97   # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.96     # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.35  # the amount of height reserved for white space between subplots
fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
matplotlib.rcParams.update({'font.size': 10})
mpl.rcParams['lines.markersize'] = 5
varfs = 11.0
nc = 1
leg_loc = 'lower right'
fs = 8.0
lc = 'lightgrey'
ls = ':'
lw = 0.50

line_color = 'black'
markers = ['^', 's', 'o', 'v', '*']
colors = ['blue', 'green', 'red', 'black', 'magenta']


# #_________________________________________________________________
# ## Storage modulus
ax1 = plt.subplot(231)
ax1.plot(www, Ep_predict, label='Predicted', color=line_color)

ax1.scatter(ww[0:row_ct[0]], Summary_Table['E\''][0:row_ct[0]], marker = markers[0], color = colors[0], label=str(round(tempavgall[0]))+' °C')
ax1.scatter(ww[row_ct[0]:sum(row_ct[0:2])], Summary_Table['E\''][row_ct[0]:sum(row_ct[0:2])], marker = markers[1], color = colors[1], label=str(round(tempavgall[1]))+' °C')
ax1.scatter(ww[sum(row_ct[0:2]):sum(row_ct[0:3])], Summary_Table['E\''][sum(row_ct[0:2]):sum(row_ct[0:3])], marker = markers[2], color = colors[2], label=str(round(tempavgall[2]))+' °C')
ax1.scatter(ww[sum(row_ct[0:3]):sum(row_ct[0:4])], Summary_Table['E\''][sum(row_ct[0:3]):sum(row_ct[0:4])], marker = markers[3], color = colors[3], label=str(round(tempavgall[3]))+' °C')
ax1.scatter(ww[sum(row_ct[0:4]):sum(row_ct[0:5])], Summary_Table['E\''][sum(row_ct[0:4]):sum(row_ct[0:5])], marker = markers[4], color = colors[4], label=str(round(tempavgall[4]))+' °C')

ax1.set_ylabel(r'$E^{\prime}$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc=leg_loc)
ax1.set_xlim(1E-5, 1E7)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Storage Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6, 1E8])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)

# #_________________________________________________________________
# ## Loss modulus
ax1 = plt.subplot(232)
ax1.plot(www, Epp_predict, label='Predicted', color=line_color)

ax1.scatter(ww[0:row_ct[0]], Summary_Table['E\"'][0:row_ct[0]], marker = markers[0], color = colors[0], label=str(round(tempavgall[0]))+' °C')
ax1.scatter(ww[row_ct[0]:sum(row_ct[0:2])], Summary_Table['E\"'][row_ct[0]:sum(row_ct[0:2])], marker = markers[1], color = colors[1], label=str(round(tempavgall[1]))+' °C')
ax1.scatter(ww[sum(row_ct[0:2]):sum(row_ct[0:3])], Summary_Table['E\"'][sum(row_ct[0:2]):sum(row_ct[0:3])], marker = markers[2], color = colors[2], label=str(round(tempavgall[2]))+' °C')
ax1.scatter(ww[sum(row_ct[0:3]):sum(row_ct[0:4])], Summary_Table['E\"'][sum(row_ct[0:3]):sum(row_ct[0:4])], marker = markers[3], color = colors[3], label=str(round(tempavgall[3]))+' °C')
ax1.scatter(ww[sum(row_ct[0:4]):sum(row_ct[0:5])], Summary_Table['E\"'][sum(row_ct[0:4]):sum(row_ct[0:5])], marker = markers[4], color = colors[4], label=str(round(tempavgall[4]))+' °C')

ax1.set_ylabel(r'$E^{\prime\prime}$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc='upper left')
ax1.set_xlim(1E-5, 1E7)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Loss Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6, 1E8])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)


# #_________________________________________________________________
# ## Phase Angle
ax1 = plt.subplot(233)

ax1.scatter(Summary_Table['Reduced_Freq'][0:row_ct[0]], Summary_Table['Phase_Angle'][0:row_ct[0]], marker = markers[0], color = colors[0], label=str(round(tempavgall[0]))+' °C')
ax1.scatter(Summary_Table['Reduced_Freq'][row_ct[0]:sum(row_ct[0:2])], Summary_Table['Phase_Angle'][row_ct[0]:sum(row_ct[0:2])], marker = markers[1], color = colors[1], label=str(round(tempavgall[1],1))+' °C')
ax1.scatter(Summary_Table['Reduced_Freq'][sum(row_ct[0:2]):sum(row_ct[0:3])], Summary_Table['Phase_Angle'][sum(row_ct[0:2]):sum(row_ct[0:3])], marker = markers[2], color = colors[2], label=str(round(tempavgall[2]))+' °C')
ax1.scatter(Summary_Table['Reduced_Freq'][sum(row_ct[0:3]):sum(row_ct[0:4])], Summary_Table['Phase_Angle'][sum(row_ct[0:3]):sum(row_ct[0:4])], marker = markers[3], color = colors[3], label=str(round(tempavgall[3]))+' °C')
ax1.scatter(Summary_Table['Reduced_Freq'][sum(row_ct[0:4]):sum(row_ct[0:5])], Summary_Table['Phase_Angle'][sum(row_ct[0:4]):sum(row_ct[0:5])], marker = markers[4], color = colors[4], label=str(round(tempavgall[4]))+' °C')

ax1.set_ylabel(r'$\phi$' + ' ' + r'$(^\circ)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc='upper right')
ax1.set_xlim(1E-5, 1E7)
ax1.set_ylim(0, 40)
ax1.set_title('Phase Angle', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6, 1E8])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)

#_________________________________________________________________
## Shift Factor
ax1 = plt.subplot(234)
ax1.scatter(tempavgall, log_at_val, color='fuchsia', marker='d')
m, b = np.polyfit(tempavgall, log_at_val, 1)
plt.plot(tempavgall, m*tempavgall + b, color = line_color, linewidth = 1.5, label='y={:.2f}x+{:.2f}'.format(m,b))
ax1.set_ylabel(r'$Log(a_T)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$T$' + ' ' + r'$(^\circ C)$', fontweight='bold', fontsize=varfs)
ax1.legend(fontsize=10.0,ncol=nc,loc='upper right')
ax1.set_xlim(-20, 60)
ax1.set_ylim(-5, 5)
ax1.set_title('Shift Factors', fontweight='bold', fontsize=varfs)
ax1.grid(color = lc, linestyle = ls, linewidth = lw)
C1_text = r'$C_1=$' + str(round(C[0],2))
C2_text = r'$C_2=$' + str(round(C[1],1))
ax1.text(-15.0, -3.2, C1_text, ha='left', rotation=0, family='sans-serif', size=9,  backgroundcolor='w', color='black')
ax1.text(-15.0, -4.0, C2_text, ha='left', rotation=0, family='sans-serif', size=9,  backgroundcolor='w', color='black')

#_________________________________________________________________    
## Predicted dynamic modulus
ax1 = plt.subplot(235)
ax1.plot(www, Estar_predict, label='Predicted', color=line_color)

ax1.scatter(ww[0:row_ct[0]], Summary_Table['Dynamic_Modulus'][0:row_ct[0]], marker = markers[0], color = colors[0], label=str(round(tempavgall[0]))+' °C')
ax1.scatter(ww[row_ct[0]:sum(row_ct[0:2])], Summary_Table['Dynamic_Modulus'][row_ct[0]:sum(row_ct[0:2])], marker = markers[1], color = colors[1], label=str(round(tempavgall[1]))+' °C')
ax1.scatter(ww[sum(row_ct[0:2]):sum(row_ct[0:3])], Summary_Table['Dynamic_Modulus'][sum(row_ct[0:2]):sum(row_ct[0:3])], marker = markers[2], color = colors[2], label=str(round(tempavgall[2]))+' °C')
ax1.scatter(ww[sum(row_ct[0:3]):sum(row_ct[0:4])], Summary_Table['Dynamic_Modulus'][sum(row_ct[0:3]):sum(row_ct[0:4])], marker = markers[3], color = colors[3], label=str(round(tempavgall[3]))+' °C')
ax1.scatter(ww[sum(row_ct[0:4]):sum(row_ct[0:5])], Summary_Table['Dynamic_Modulus'][sum(row_ct[0:4]):sum(row_ct[0:5])], marker = markers[4], color = colors[4], label=str(round(tempavgall[4]))+' °C')

ax1.set_ylabel(r'$|E^*|$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc=leg_loc)

#print the value of E_0
E_0_text = r'$E_0=$' + str(round(E_0,1)) + ' ' + f'MPa'
ax1.text(1E-3, 4E4, E_0_text, ha='left', rotation=0, family='sans-serif', size=9,  backgroundcolor='w', color='black')
ax1.set_xlim(1E-5, 1E7)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Dynamic Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6, 1E8])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)

#_________________________________________________________________
## Predicted dynamic modulus
ax1 = plt.subplot(236)
ax1.plot(ttt, Estar_predict, label='Predicted', color=line_color)
ax1.set_ylabel(r'$E(t)$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$t$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc=leg_loc)
ax1.set_xlim(1E-5, 1E7)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Dynamic Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6, 1E8])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)

plt.tight_layout()
plt.show()

fig.savefig('Dynamic_Mod_'+case+'.png',dpi=400)

# Measure end run time
end = time.time()

# total time taken
print(f"Runtime: {round(end - start,4)} seconds.")
# %%
