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
#              Time History Plotting Script 
#                  By: Johann J Cardenas
# '----------------'  '----------------'  '----------------' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import os

##################################################
########            User Inputs           ######## 
##################################################

CaseList = {'CC71DS_P5_AC1S_B1_SB1_SG1': list(range(1, 16)),
            'CC72DS_P4_AC1S_B1_SB1_SG1': list(range(1, 29)),
            'CC73DS_P5_AC1S_B1_SB1_SG1': list(range(1, 40)),}

# Loading Nomenclature
# P1 = 34.5 kip
# P2 = 35.5 kip
# P3 = 37.0 kip
# P4 = 38.0 kip
# P5 = 39.0 kip
# P6 = 40.5 kip
# P7 = 45.0 kip
# P8 = 46.0 kip
# P9 = 55.0 kip
# P10 = 65.0 kip

# Model Dimensions (Update per case)
L =  [32677.0, 34120.0,  35507.0]        # Length of the Model
Xw = [1677.0, 3120.0, 4507.0]            # Length of the Wheel Path
B =  [32750.0, 32750.0,  32750.0]        # Width of the Model
b = [1750.0, 1750.0, 1750.0]             # Width of the Wheel Path
Depth = [15000.0,  15000.0,  15000.0]    # Total Depth[mm] of the Model

Structure = ['AC1', 'B1', 'SB1', 'SG1']   # Pavement Layers
MyLabels = ['P-401', 'P-209', 'P-154', 'Subgrade'] # Plot Labels
Thicks = [75.0, 150.0, 500.0, 14275.0]    # Thickness of each layer

user = 'johan'
directory = f'C:/Users/{user}/Box/FAA Data Project/04_FEM/00_FEM DATA/FAA_South/FAA_South_Responses/'


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
########           STRAIN PLOTS           ######## 
##################################################

plt.rcParams.update({'font.size': 12})


locations = []
labels = ['Single, 39 kips', 'Tandem, 38 kips', 'Tridem, 39 kips']
markers = ['o', 's', 'D']
colors = ['r', 'b', 'g']

selected_ly = 'AC1'
loc_rel = [1443.0, 50.0 , 730.0 ]

# Longitudinal Profile Along the Z Axis
def plot_ETH(field, myData):
    strain_values = []
    for _ in CaseList.keys():
        strain_values.append([])
         
    for case_idx, (case, values) in enumerate(CaseList.items()): 
        location = [(L[case_idx]/2 - Xw[case_idx]/2) + loc_rel[0], 
                    Depth[case_idx] - loc_rel[1], 
                    B[case_idx]/2 - loc_rel[2]]
        locations.append(location)
        
        for ts in values:
            data_ts = myData[case_idx][selected_ly][ts-1].copy()  # Access 'AC1' layer data for this tire state
            
            distances = np.sqrt((data_ts['Xn_elem'] - location[0]) ** 2 + 
                                (data_ts['Yn_elem'] - location[1]) ** 2 + 
                                (data_ts['Zn_elem'] - location[2]) ** 2)
            idx_min = distances.idxmin()  # Index of the closest point
            data_ts = data_ts.loc[[idx_min]]

            if field == 'U2':
                label = r'Deflection $U_2$ (mm)'
                strain_value = data_ts[field].iloc[0]

            elif field == 'E11':
                label = r"Longitudinal Tensile Strain $E_{11}$ ($\mu\epsilon$)"
                strain_value = data_ts[field].iloc[0]*1000000
                
            elif field == 'E22':
                label = r"Vertical Compressive Strain $E_{22}$ ($\mu\epsilon$)"
                strain_value = data_ts[field].iloc[0]*1000000
                
            elif field == 'E23':
                label = r"Vertical Shear Strain $E_{23}$ ($\mu\epsilon$)"
                strain_value = data_ts[field].iloc[0]*1000000
            
            elif field == 'E33':
                label = r"Transverse Tensile Strain $E_{33}$ ($\mu\epsilon$)"
                strain_value = data_ts[field].iloc[0]*1000000
                
            strain_values[case_idx].append(strain_value)
            
        
    fig, ax = plt.subplots(figsize=(6,5), dpi=300)
    for i, case in enumerate(CaseList.keys()):
        ax.plot(strain_values[i], color=colors[i], marker=markers[i], 
                markersize=5, linestyle='-',linewidth=2, label=labels[i])
    
    ax.set_ylabel(label, fontweight='bold', fontsize=14)
    ax.set_xlabel('Time Step', fontweight='bold', fontsize=14)
    ax.set_xlim(0, len(CaseList[case]))
    ax.legend()
    
    plt.savefig(f'{selected_ly}_{field}_TimeHistory.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()
    

    
def plot_EMax(fields, myData):
    # Assuming 'selected_ly' is defined
    selected_ly = 'AC1'
    label_selected_ly = 'Asphalt Layer'
    # Assuming 'colors' is defined with enough colors for your fields
    colors = ['r', 'g', 'b', 'y']  # Example: Different color for each field
    labels = [r"$E_{22}$",
              r"$E_{11}$",
              r"$E_{33}$",
              r"$E_{23}$"]

    legend_labels = ['Single, 39 kips', 'Tandem, 38 kips', 'Tridem, 39 kips']
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    width = 0.30  # Bar width
    num_cases = 3  # Number of cases per field
    num_fields = len(fields)

    for field_idx, field in enumerate(fields):
        
        for case_idx, (case, values) in enumerate(CaseList.items()):
            max_strains = []

            for ts in values:
                data_ts = myData[case_idx][selected_ly][ts-1].copy()
                # Initialize strain_value at a safe default, if necessary
                strain_value = 0  # Default value, in case the field does not match
                
                if field == 'U2':
                    label = r'Deflection $U_2$ (mm)'
                    strain_value = data_ts[field].max()
                elif field in ['E22', 'E11', 'E33', 'E23']:
                    if field == 'E11':
                        strain_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                    elif field == 'E22':
                        strain_value = abs(data_ts[field].min()) * 1000000  # Absolute value of the minimum
                    elif field == 'E33':
                        strain_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                    elif field == 'E23':
                        max_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                        min_value = abs(data_ts[field].min()) * 1000000  # Absolute value of the minimum
                        strain_value = max(max_value, min_value)  # Maximum between the absolute values of maximum and minimum
                
                max_strains.append(strain_value)
            
            # Calculate the position for each bar to create a group effect
            positions = [x + (case_idx - (num_cases / 2)) * width for x in range(num_fields)]
            # Plot
            ax.bar(positions[field_idx], max(max_strains), width, label=f'{case}', color=colors[case_idx])

    ax.set_ylabel(r"Maximum Strain ($\mu\epsilon$)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Strain Fields", fontweight='bold', fontsize=14)
    ax.set_xticks([i-width/2 for i in range(num_fields)])  # Center x-ticks between groups
    ax.set_xticklabels(labels, fontsize=14)  # Use custom labels
    ax.legend(fontsize=12, labels=legend_labels)
    ax.title.set_text(f'{label_selected_ly}')
    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:num_cases], legend_labels[:num_cases], fontsize=10)

    plt.savefig(f'{selected_ly}_MaxStrains.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


##################################################
########             PLOTS                ######## 
##################################################

#plot_ETH('U2', myData_layers)
#plot_ETH('E22', myData_layers)
#plot_ETH('E11', myData_layers)
#plot_ETH('E33', myData_layers)
#plot_ETH('E23', myData_layers)


# Define fields to plot
plot_EMax(['E22', 'E11', 'E33', 'E23'], myData_layers) 