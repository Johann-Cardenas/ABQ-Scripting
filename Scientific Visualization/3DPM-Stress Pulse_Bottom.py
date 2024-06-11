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
#             Stress Pulse Analysis (Dynamic Modulus)
#                  By: Johann J Cardenas
# '----------------'  '----------------'  '----------------' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os

##################################################
########            User Inputs           ######## 
##################################################

CaseList = {'LV_P1_SL0': list(range(1, 19)),
            'LV_P1_SL2': list(range(1, 19)),
            'LV_P2_SL0': list(range(1, 19)),
            'LV_P2_SL2': list(range(1, 19)),
            'LV_P3_SL0': list(range(1, 19)),
            'LV_P4_SL0': list(range(1, 19)),
            'LV_P4_SL6': list(range(1, 19)),
            'LV_P5_SL0': list(range(1, 19)),
            'LV_P5_SL3': list(range(1, 19)),
            'LV_P6_SL0': list(range(1, 19)),
            'LV_P6_SL6': list(range(1, 19))}

# Model Dimensions  # LV_P1_SL0
L =  [9520.0,
      9540.0,
      9480.0,
      9480.0,
      9600.0,
      9480.0,
      9480.0,
      9540.0,
      9560.0,
      9480.0,
      9480.0]        # Length of the Model

Xw = [1320.0,
      1340.0,
      1280.0,
      1280.0,
      1400.0,
      1280.0,
      1280.0,
      1340.0,
      1360.0,
      1280.0,
      1280.0]            # Length of the Wheel Path

B =  [8432.0,
      8433.4,
      8802.8,
      8805.0,
      8432.9,
      8801.6,
      8807.2,
      8431.8,
      8435.3,
      8802.8,
      8806.6]        # Width of the Model

b =  [232.0,
      233.4,
      602.8,
      605.0,
      232.9,
      601.6,
      607.2,
      231.8,
      235.3,
      602.8,
      606.6]             # Width of the Wheel Path

Depth = [5000.0, 
         5000.0,
         5000.0,
         5000.0,
         5000.0,
         5000.0,
         5000.0,
         5000.0,
         5000.0,
         5000.0,
         5000.0]    # Total Depth[mm] of the Model

Structure = ['Surf','AC1', 'AC2', 'B1', 'SB1', 'SG1']   # Pavement Layers
MyLabels = ['Surf','AC1', 'AC2', 'Base','Subbase', 'Subgrade'] # Plot Labels
Thicks = [5.0, 35.0, 55.0, 205.0, 150.0, 4550.0]    # Thickness of each layer

ti =  0.0432  # Time Increment

user = 'johan'
directory = f'C:/Users/{user}/Box/R27-252 EV/Tasks/Task 3 - Pavement FEM/Post-Processing/'

##################################################
########     Preliminary Calculations     ######## 
##################################################

myData_layers = []  # Outer list to contain data for all cases
myData_cases = []   # Outer list to store case-specific data

for case in CaseList.keys():
    case_layers = {layer: [] for layer in Structure}  # A dictionary for each case, keys being layers
    myData_layers.append(case_layers)
    myData_cases.append([])  # Still keeping it for storing case-specific data without layer distinction

y_ranges = {}
cumulative_thickness = 0
for layer, thickness in zip(Structure, Thicks):
    y_ranges[layer] = (Depth[0] - cumulative_thickness - thickness, Depth[0] - cumulative_thickness)
    cumulative_thickness += thickness

##################################################
########     SEGMENTATION FUNCTION        ######## 
##################################################

def process_layers(dataf):
    
    dataframes = []
    for _ in Structure:
        # Initialize an empty DataFrame for each layer
        dataframes.append(pd.DataFrame())

    for i, lay in enumerate(Structure):
        y_lower, y_upper = y_ranges[lay]
        dfl = dataf[dataf['Yn_elem'].between(y_lower, y_upper)]
        
        if i == 0:  # First layer (top)
            dataframes[i] = dfl.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='first')
        elif i == len(Structure) - 1:  # Last layer (bottom)
            dataframes[i] = dfl.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
        else:  # Intermediate layers
            df_int_high = dfl[dfl['Yn_elem'] == y_upper].drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
            df_int_low = dfl[dfl['Yn_elem'] == y_lower].drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='first')
            dataframes[i] = pd.concat([dfl[(dfl['Yn_elem'] > y_lower) & (dfl['Yn_elem'] < y_upper)], df_int_low, df_int_high])
    
    return dataframes

##################################################

k = 0
for case, values in CaseList.items():   
    for ts in values:
        filename = f'{case}_3DResponse_tire{ts}.txt'
        filepath = os.path.join(directory + case + '/', filename)
        
        # Reading the file for each tire state
        df = pd.read_csv(filepath, sep='\t')
        data_case = df.copy()
        data_case = data_case.sort_values(by='Node', ascending=True)
        
        # Process data for each layer
        data_layers = process_layers(data_case)  # Utilize the existing 'process_layers' function
        
        for layer_name, data_layer in zip(Structure, data_layers):
            myData_layers[k][layer_name].append(data_layer)  # Append processed layer data to the corresponding case and layer
        
        myData_cases[k].append(data_case)  # Append the unsegmented case data as before
    
    k += 1
    
#________________________________________________________________________________________
# Break to start plotting function
#%% 

##################################################
########       STRESS PULSE PLOT          ######## 
##################################################

plt.rcParams.update({'font.size': 12})

locations = []
labels = ['4.7 kips, SL=0%']
markers = ['o', 's', 'D']
colors = plt.cm.GnBu(np.linspace(0.50, 1, len(L)+1))

loc_rel = [95.0]   # Depth of Analysis
dta_var = [232.00/2,
           233.40/2,
           233.40/2,
           234.50/2,
           232.90/2,
           232.80/2,
           235.60/2,
           231.80/2,
           235.30/2,
           233.40/2,
           235.30/2]

def haversine(x, b1, b2, b3, theta=np.pi/2, power=1000):
     return np.sin(theta + (np.abs(x + b3) ** b1) / b2) ** power  # Revised haversine
            
# Longitudinal Profile Along the Z Axis
def plot_SP(field, myData, selected_ly):
    
    stress_values = []
    locations = []
    
    for _ in CaseList.keys():
        stress_values.append([])
         
    for case_idx, (case, values) in enumerate(CaseList.items()): 
        location = [(L[case_idx]/2 - Xw[case_idx]/2) + Xw[case_idx]/2, 
                    Depth[case_idx] - loc_rel[0], 
                    (B[case_idx]/2 - b[case_idx]/2) + dta_var[case_idx]]
        locations.append(location)
        
        for ts in values:
            data_ts = myData[case_idx][selected_ly][ts-1].copy()  # Access layer data for this tire state
            
            distances = np.sqrt((data_ts['Xn_elem'] - location[0]) ** 2 + 
                                (data_ts['Yn_elem'] - location[1]) ** 2 + 
                                (data_ts['Zn_elem'] - location[2]) ** 2)
            idx_min = distances.idxmin()  # Index of the closest point
            data_ts = data_ts.loc[[idx_min]]

            if field == 'S22':
                label = r'Normalized Stress $S_{22}$ (MPa)'
                stress_value = data_ts[field].iloc[0]
                
            stress_values[case_idx].append(stress_value)
                    
    for i, case in enumerate(CaseList.keys()):
        fig, ax = plt.subplots(figsize=(6,5), dpi=300)
        x_data = np.arange(len(stress_values[i]))
        print(x_data)
        y_data = np.array(stress_values[i])
        print(y_data)
        
        # Normalize data to range [0, 1]
        y_data_norm = (y_data) / (np.min(y_data))
        print(y_data_norm)
        
        ax.scatter(x_data, y_data_norm, color=colors[1], marker=markers[i], s=30, label='Normalized Data')
        
        # Fitting the haversine function
        initial_guess = [1.0, 10.0, 0.0]
        bounds = ([0, 0, -np.inf], [10, 100, np.inf])
        
        try:
            popt, _ = curve_fit(lambda x, b1, b2, b3: haversine(x, b1, b2, b3, power=1000), x_data, y_data_norm, p0=initial_guess, bounds=bounds, maxfev=10000)
            
            # Calculate the pulse duration
            tex = 10   # Extending time steps
            x_fit = np.linspace(x_data[0], x_data[-1]+ tex, 1000)
            y_fit = haversine(x_fit, *popt, power=1000)
            
            Y_min = 0.01
            T = np.abs(y_fit - Y_min)
            X0 = x_fit[np.argmin(T)]
            Xt = x_fit[np.argmax(T)]
            t_RH = np.abs(Xt - X0) * 2
            print(f'{case}: t_RH={t_RH:.3f}, b1={popt[0]:.3f}, b2={popt[1]:.3f}, b3={popt[2]:.3f}')
            
            # Calculate MSE, MSRE, and R2 of the fit
            x_fit_orig = x_data
            y_fit_orig = haversine(x_fit_orig, *popt, power=1000)
            MSE = np.mean((y_data_norm - y_fit_orig) ** 2)
            MSRE = np.sqrt(MSE)
            R2 = 1 - np.sum((y_data_norm - y_fit_orig) ** 2) / np.sum((y_data_norm - np.mean(y_data_norm)) ** 2)
            print(f'{case}: MSE={MSE:.3f}, MSRE={MSRE:.3f}, R2={R2:.3f}')

            # Compute period and frequency
            peak_value = y_fit[np.argmax(y_fit)]
            init_value = y_data_norm[0]
            
            ppeak_val = y_fit[np.argmax(y_fit):]
            post_peak_x = x_fit[np.argmax(y_fit):]
            
            close_idx = np.argmin(np.abs(ppeak_val - init_value))
            matching_x = post_peak_x[close_idx]
            matching_y = ppeak_val[close_idx]
            
            tsteps = matching_x - x_fit[0]
            print(f'{case}: N={tsteps:.2f} tsteps, t={tsteps*ti:.3f} sec, f={1/(2*np.pi*tsteps*ti):.3f} Hz')

            ax.plot(x_fit, y_fit, color=colors[0], linestyle='--', linewidth=1.25, label='Haversine Fit')
            ax.scatter([x_fit[0], matching_x], [y_fit[0], matching_y], color=colors[0],  marker=markers[1], s=30)
            
        except RuntimeError as e:
            print(f"Error fitting data for {case}: {e}")
            
        ax.set_ylabel(label, fontweight='bold', fontsize=14)
        ax.set_xlabel('Time Step', fontweight='bold', fontsize=14)
        ax.set_xlim(0, len(CaseList[case]) + 10)
        ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.set_title('Stress Pulse', fontweight='bold', fontsize=18)
        
        plt.savefig(f'{selected_ly}_{field}_Stress_Pulse.png', dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

##################################################
########             PLOTS                ######## 
##################################################

plot_SP('S22', myData_layers, 'AC2')
