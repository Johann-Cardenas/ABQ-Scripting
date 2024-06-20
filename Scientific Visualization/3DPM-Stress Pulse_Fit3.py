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

CaseList = {'FD_P1_SL0': list(range(1, 19))}

# Model Dimensions  # LV_P1_SL0
L =  [9520.0]        # Length of the Model

Xw = [1320.0]            # Length of the Wheel Path

B =  [8432.0]        # Width of the Model

b =  [232.0]             # Width of the Wheel Path

Depth = [5000.0]    # Total Depth[mm] of the Model

# FULL DEPTH
Structure = ['Dummy','Surf','AC1', 'AC2', 'AC3','B1', 'SG1']   # Pavement Layers
MyLabels = ['Dummy','Surf','AC1', 'AC2', 'AC3', 'Base', 'Subgrade'] # Plot Labels
Thicks = [6.10, 6.14, 37.76, 55.0, 145.0, 305.0, 4445.0]    # Thickness of each layer

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
markers = ['o', 's', 'D'] 

# Define color palette
color_hex = ['043353', 'E44652', 'FA3283' , '00C7C7', '4D4957']

def hex_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove '#' if it's there
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))  # Normalize to [0, 1] range

colors = [hex_rgb(color) for color in color_hex]

loc_rel = [250.0]   # Depth of Analysis
dta_var = [232.00/2]
dh = 400.0   # Distance of the analysis point from the middle of the wheelpath

# Longitudinal Profile Along the Z Axis
def fit3(field, myData, selected_ly):
    
    def new_function(t, s):
        return np.exp(-t**2 / s**2)
    
    stress_values = []
    locations = []
    
    for _ in CaseList.keys():
        stress_values.append([])
         
    for case_idx, (case, values) in enumerate(CaseList.items()): 
        location = [(L[case_idx]/2 - Xw[case_idx]/2) + Xw[case_idx]/2 + dh, 
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
                label = r'Normalized Stress $S_{22}$'
                stress_value = data_ts[field].iloc[0]
                
            stress_values[case_idx].append(stress_value)
                    
    for i, case in enumerate(CaseList.keys()):
        fig, ax = plt.subplots(figsize=(6,4), dpi=300)
        x_data = np.arange(len(stress_values[i]))
        print(f'Time Steps: {x_data}')
        x_data = x_data*ti
        print(f'Time Domain: {x_data}')
        y_data = np.array(stress_values[i])
        print(f'Raw Stress Values: {y_data}')

        # Normalize data to range [0, 1]
        y_data_norm = (y_data) / (np.min(y_data))
        print(f'Normalized Data: {y_data_norm}')
        
        peak_idx = np.argmax(y_data_norm)
        x_data= x_data - x_data[peak_idx]
        print(f'Centered Time Domain: {x_data}')
        
        ax.scatter(x_data, y_data_norm, color=colors[0], marker=markers[0], s=25, label='Normalized Data')
        
        # Calculate 'd' based on peak value of 'y'
        peak_idx = np.argmax(y_data_norm)
        tex = abs(x_data[peak_idx]-x_data[0]) # Extending time steps

        s_initial = np.std(x_data)
   
        # Fitting the new function
        try:
            popt, _ = curve_fit(new_function, x_data, y_data_norm, p0=[s_initial], bounds=(0, np.inf), maxfev=10000)
            
            # Calculate the pulse duration
            x_fit = np.linspace(x_data[0], x_data[-1]+ tex, 1000)
            y_fit = new_function(x_fit, *popt)
            
            # Calculate MSE, MSRE, and R2 of the fit
            x_fit_orig = x_data
            y_fit_orig = new_function(x_fit_orig, *popt)
            MSE = np.mean((y_data_norm - y_fit_orig) ** 2)
            MSRE = np.sqrt(MSE)
            R2 = 1 - np.sum((y_data_norm - y_fit_orig) ** 2) / np.sum((y_data_norm - np.mean(y_data_norm)) ** 2)
            print(f'Error Metrics {case}: MSE={MSE:.3f}, MSRE={MSRE:.3f}, R2={R2:.3f}')

            # Compute period and frequency
            peak_index = np.argmax(y_fit)
            peak_value = y_fit[peak_index]
            #init_value = y_data_norm[0]
            init_value = y_fit[0]
            
            # Extract post-peak values
            post_peak_y = y_fit[peak_index:]
            post_peak_x = x_fit[peak_index:]
            
            close_idx = np.argmin(np.abs(post_peak_y - init_value))
            matching_x = post_peak_x[close_idx]
            matching_y = post_peak_y[close_idx]
                        
            tsteps = matching_x - x_fit[0]
            print(f'{case}: t={tsteps:.3f} sec, f={1/(2*np.pi*tsteps):.3f} Hz')

            ax.plot(x_fit, y_fit, color=colors[3], linestyle='-', linewidth=1.25, label=f'Normalized Bell, $R^2$:{R2:.3f} ')
            ax.scatter([x_fit[0], matching_x], [y_fit[0], matching_y], color=colors[3],  marker=markers[1], s=25)
            
        except RuntimeError as e:
            print(f"Error fitting data for {case}: {e}")
            
        ax.set_ylabel(label, fontweight='bold', fontsize=14)
        ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
        ax.set_xlim(x_data[0], x_data[-1] + tex)
        ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_title('Stress Pulse', fontweight='bold', fontsize=18)
        
        # plt.savefig(f'{case}_Stress_Pulse.png', dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

##################################################
########             PLOTS                ######## 
##################################################

fit3('S22', myData_layers, 'AC3')
