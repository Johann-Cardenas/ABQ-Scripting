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
case = 'SMA_T1-80-12.5-0 (A)'
row_ct = [4, 4, 5]              # number of frequencies tested per test temp
Test_tempF = [39.2, 68.0, 113.0] # degF   [4.0, 20.0, 45.0] degC

shift_factors = [19.25, 0, 24.25] # Guesstimate of Shift factors; 0 for reference temp [values should be positive]
rr = np.logspace(-5,5,16)     # seed values for Prony series fitting; initial assumptions are important; check final plot
tref = Test_tempF[1]
To = (tref-32.)*5./9.           # Reference temp, degC
rheo_factor = 2*np.pi           # 1  vs 2*np.pi
Einf = 1.0                      # Long-term modulus, MPa

Test_temp_all = np.concatenate((np.ones(row_ct[0])*Test_tempF[0],
                                np.ones(row_ct[1])*Test_tempF[1],
                                np.ones(row_ct[2])*Test_tempF[2]))
# Unit conversion
psi_mpa = 6.8947572932 #  6.89 MPa = 1 ksi
tempC= [(i-32.)*5./9. for i in Test_tempF] ## degC
Test_tempC = [(i-32.)*5./9. for i in Test_temp_all] ## degC

## Units: Test_Temp (degF), Estar (ksi), Delta (degrees)

data = {'Test_TempF': Test_temp_all,
        'Test_TempC': Test_tempC,
        'Frequency':[25.0,  10.0,  1.0,  0.1,    # test temp = 39.2 degF [4.0 degC]
                     25.0,  10.0,  1.0,  0.1,    # test temp = 68 degF [20.0 degC]
                     25.0,  10.0,  1.0,  0.1, 0.01],   # test temp = 113 degF [45.0 degC]
        'Estar1':[10314.,  8858.,  5661.,  3263.,   # MPa, A1
                  4165.,  3232.,  1516.,  637.8,    
                  700.,  445.3,  198.4,  121.4,  92.4],   
        'Estar2':[9749.,  8488.,  5251.,  2959.,   # MPa, A3
                  4042.,  3122.,  1465.,  617.,  
                  544.7,  334.0,  145.9,  88.9,  70.5],  
        'Delta1':[15.18,  16.78,  21.04,  25.19,      # degrees
                  26.47,  28.18,  31.23,  31.38,
                  31.58,  30.95,  24.77,  18.33,  14.05],
        'Delta2':[15.46,  17.09,  21.43,  26.38,      # degrees
                  26.32,  28.15,  31.46,  31.92, 
                  35.64,  36.25,  30.02,  22.78,  17.78]}


# Create dataframe
Summary_Table = pd.DataFrame(data)

#Estar_avg = psi_mpa * (Summary_Table['Estar1'] + Summary_Table['Estar2']) / 2.0 ## MPa
Estar_avg = (Summary_Table['Estar1'] + Summary_Table['Estar2']) / 2.0 ## MPa

Delta_avg = (Summary_Table['Delta1'] + Summary_Table['Delta2']) / 2.0 ## degrees

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
xfit = np.arange(0.5, 4.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "b^", label="Test Data" + " " + str(round(tempC[0],1))+" °C")
plt.plot(xfit, yfit, "b--", label="Fitted Model" + " " + str(round(tempC[0],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')
# plt.show()

i = 1
x = Summary_Table['Log_reduced_freq'][row_ct[0]:sum(row_ct[0:2])]
y = Summary_Table['Dynamic_Modulus'][row_ct[0]:sum(row_ct[0:2])]
xfit = np.arange(-1.0, 2.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "gs", label="Test Data" + " " + str(round(tempC[1],1))+" °C")
plt.plot(xfit, yfit, "g--", label="Fitted Model" + " " + str(round(tempC[1],1))+" °C")
plt.yscale('log')
plt.legend(loc='lower right')
# plt.show()

i = 2
x = Summary_Table['Log_reduced_freq'][sum(row_ct[0:2]):sum(row_ct)]
y = Summary_Table['Dynamic_Modulus'][sum(row_ct[0:2]):sum(row_ct)]
xfit = np.arange(-5.0, 1.0, 0.01)
yfit = fitting_model(x, y, xfit)

plt.plot(x, y, "ro", label="Test Data" + " " + str(round(tempC[2],1))+" °C")
plt.plot(xfit, yfit, "r--", label="Fitted Model" + " " + str(round(tempC[2],1))+" °C")
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
www = np.logspace(-6,6,200) #
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

# #_________________________________________________________________
# ## Storage modulus
ax1 = plt.subplot(231)
ax1.plot(www, Ep_predict, label='Predicted', color=line_color)
ax1.scatter(ww[0:row_ct[0]], Summary_Table['E\''][0:row_ct[0]], marker='^', color = 'blue', label=str(round(tempavgall[0],1))+' °C')
ax1.scatter(ww[row_ct[0]:sum(row_ct[0:2])], Summary_Table['E\''][row_ct[0]:sum(row_ct[0:2])], marker='s', color = 'green', label=str(round(tempavgall[1],1))+' °C')
ax1.scatter(ww[sum(row_ct[0:2]):sum(row_ct)], Summary_Table['E\''][sum(row_ct[0:2]):sum(row_ct)], marker='o', color = 'red', label=str(round(tempavgall[2],1))+' °C')
ax1.set_ylabel(r'$E^{\prime}$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc=leg_loc)
ax1.set_xlim(1E-4, 1E4)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Storage Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)

# #_________________________________________________________________
# ## Loss modulus
ax1 = plt.subplot(232)
ax1.plot(www, Epp_predict, label='Predicted', color=line_color)
ax1.scatter(ww[0:row_ct[0]], Summary_Table['E\"'][0:row_ct[0]], marker='^', color = 'blue', label=str(round(tempavgall[0],1))+' °C')
ax1.scatter(ww[row_ct[0]:sum(row_ct[0:2])], Summary_Table['E\"'][row_ct[0]:sum(row_ct[0:2])], marker='s', color = 'green', label=str(round(tempavgall[1],1))+' °C')
ax1.scatter(ww[sum(row_ct[0:2]):sum(row_ct)], Summary_Table['E\"'][sum(row_ct[0:2]):sum(row_ct)], marker='o', color = 'red', label=str(round(tempavgall[2],1))+' °C')
ax1.set_ylabel(r'$E^{\prime\prime}$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc='upper left')
ax1.set_xlim(1E-4, 1E4)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Loss Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)


# #_________________________________________________________________
# ## Phase Angle
ax1 = plt.subplot(233)
ax1.scatter(Summary_Table['Reduced_Freq'][0:row_ct[0]], Summary_Table['Phase_Angle'][0:row_ct[0]], marker='^', color = 'blue', label=str(round(tempavgall[0],1))+' °C')
ax1.scatter(Summary_Table['Reduced_Freq'][row_ct[0]:sum(row_ct[0:2])], Summary_Table['Phase_Angle'][row_ct[0]:sum(row_ct[0:2])], marker='s', color = 'green', label=str(round(tempavgall[1],1))+' °C')
ax1.scatter(Summary_Table['Reduced_Freq'][sum(row_ct[0:2]):sum(row_ct)], Summary_Table['Phase_Angle'][sum(row_ct[0:2]):sum(row_ct)], marker='o', color = 'red', label=str(round(tempavgall[2],1))+' °C')
ax1.set_ylabel(r'$\phi$' + ' ' + r'$(^\circ)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc='upper right')
ax1.set_xlim(1E-5, 1E4)
ax1.set_ylim(0, 40)
ax1.set_title('Phase Angle', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6])
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
ax1.set_xlim(0, 50)
ax1.set_ylim(-3, 3)
ax1.set_title('Shift Factors', fontweight='bold', fontsize=varfs)
ax1.grid(color = lc, linestyle = ls, linewidth = lw)
C1_text = r'$C_1=$' + str(round(C[0],2))
C2_text = r'$C_2=$' + str(round(C[1],1))
ax1.text(3.0, -1.8, C1_text, ha='left', rotation=0, family='sans-serif', size=9,  backgroundcolor='w', color='black')
ax1.text(3.0, -2.6, C2_text, ha='left', rotation=0, family='sans-serif', size=9,  backgroundcolor='w', color='black')

#_________________________________________________________________    
## Predicted dynamic modulus
ax1 = plt.subplot(235)
ax1.plot(www, Estar_predict, label='Predicted', color=line_color)
ax1.scatter(ww[0:row_ct[0]], Summary_Table['Dynamic_Modulus'][0:row_ct[0]], marker='^', color = 'blue', label=str(round(tempavgall[0],1))+' °C')
ax1.scatter(ww[row_ct[0]:sum(row_ct[0:2])], Summary_Table['Dynamic_Modulus'][row_ct[0]:sum(row_ct[0:2])], marker='s', color = 'green', label=str(round(tempavgall[1],1))+' °C')
ax1.scatter(ww[sum(row_ct[0:2]):sum(row_ct)], Summary_Table['Dynamic_Modulus'][sum(row_ct[0:2]):sum(row_ct)], marker='o', color = 'red', label=str(round(tempavgall[2],1))+' °C')
ax1.set_ylabel(r'$|E^*|$' + ' ' + r'$(MPa)$', fontweight='bold', fontsize=varfs)
ax1.set_xlabel(r'$\xi$', fontweight='bold', fontsize=varfs)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(fontsize=fs,ncol=nc,loc=leg_loc)

#print the value of E_0
E_0_text = r'$E_0=$' + str(round(E_0,1)) + ' ' + f'MPa'
ax1.text(1E-3, 1E4, E_0_text, ha='left', rotation=0, family='sans-serif', size=9,  backgroundcolor='w', color='black')
ax1.set_xlim(1E-4, 1E4)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Dynamic Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6])
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
ax1.set_xlim(1E-4, 1E4)
ax1.set_ylim(1E1, 1E5)
ax1.set_title('Dynamic Modulus', fontweight='bold', fontsize=varfs)
ax1.set_xticks([1E-4, 1E-2, 1E0, 1E2, 1E4])
ax1.grid(color = lc, linestyle = ls, linewidth = lw)

plt.tight_layout()
plt.show()

fig.savefig('Dynamic_Mod_'+case+'.png',dpi=400)

# Measure end run time
end = time.time()

# total time taken
print(f"Runtime: {round(end - start,4)} seconds.")