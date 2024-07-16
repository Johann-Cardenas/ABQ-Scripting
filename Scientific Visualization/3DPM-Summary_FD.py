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

CaseList = {'FD_P1_SL0': list(range(1, 19)),
            'FD_P1_SL2': list(range(1, 19)),
            'FD_P2_SL0': list(range(1, 19)),
            'FD_P2_SL2': list(range(1, 19)),
            'FD_P3_SL0': list(range(1, 19)),
            'FD_P4_SL0': list(range(1, 19)),
            'FD_P4_SL6': list(range(1, 19)),
            'FD_P5_SL0': list(range(1, 19)),
            'FD_P5_SL3': list(range(1, 19)),
            'FD_P6_SL0': list(range(1, 19)),
            'FD_P6_SL6': list(range(1, 19))}

# Model Dimensions (Update per case)
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

Structure = ['Dummy','Surf','AC1', 'AC2', 'AC3','B1', 'SG1']   # Pavement Layers
MyLabels = ['Dummy','Surf','AC1', 'AC2', 'AC3', 'Base', 'Subgrade'] # Plot Labels
Thicks = [6.10, 6.14, 37.76, 55.0, 145.0, 305.0, 4445.0]    # Thickness of each layer

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
########           STRAIN PLOTS           ######## 
##################################################

plt.rcParams.update({'font.size': 12})


locations = []
labels = ['6.0 kips, SL=0%', 
          '6.0 kips, SL=2%', 
          '4.2 kips, SL=0%', 
          '4.2 kips, SL=2%', 
          '9.5 kips, SL=0%', 
          '4.7 kips, SL=0%', 
          '4.7 kips, SL=6%', 
          '7.0 kips, SL=0%', 
          '7.0 kips, SL=3%', 
          '4.2 kips, SL=0%', 
          '4.2 kips, SL=6%']




markers = ['o', 
           'o',
           'o',
           'o',
           'o',
           'o',
           'o',
           'o',
           'o',
           'o',
           'o']

colors = plt.cm.GnBu(np.linspace(0.1, 1, 11))

# loc_rel = [1443.0, 75.0 , 675.0 ]

# # Longitudinal Profile Along the Z Axis
# def plot_ETH(field, myData):
#     strain_values = []
#     for _ in CaseList.keys():
#         strain_values.append([])
         
#     for case_idx, (case, values) in enumerate(CaseList.items()): 
#         location = [(L[case_idx]/2 - Xw[case_idx]/2) + loc_rel[0], 
#                     Depth[case_idx] - loc_rel[1], 
#                     B[case_idx]/2 - loc_rel[2]]
#         locations.append(location)
        
#         for ts in values:
#             data_ts = myData[case_idx][selected_ly][ts-1].copy()  # Access 'AC1' layer data for this tire state
            
#             distances = np.sqrt((data_ts['Xn_elem'] - location[0]) ** 2 + 
#                                 (data_ts['Yn_elem'] - location[1]) ** 2 + 
#                                 (data_ts['Zn_elem'] - location[2]) ** 2)
#             idx_min = distances.idxmin()  # Index of the closest point
#             data_ts = data_ts.loc[[idx_min]]

#             if field == 'U2':
#                 label = r'Deflection $U_2$ (mm)'
#                 strain_value = data_ts[field].iloc[0]

#             elif field == 'E11':
#                 label = r"Longitudinal Tensile Strain $E_{11}$ ($\mu\epsilon$)"
#                 strain_value = data_ts[field].iloc[0]*1000000
                
#             elif field == 'E22':
#                 label = r"Vertical Compressive Strain $E_{22}$ ($\mu\epsilon$)"
#                 strain_value = data_ts[field].iloc[0]*1000000
                
#             elif field == 'E23':
#                 label = r"Vertical Shear Strain $E_{23}$ ($\mu\epsilon$)"
#                 strain_value = data_ts[field].iloc[0]*1000000
            
#             elif field == 'E33':
#                 label = r"Transverse Tensile Strain $E_{33}$ ($\mu\epsilon$)"
#                 strain_value = data_ts[field].iloc[0]*1000000
                
#             strain_values[case_idx].append(strain_value)
            
        
#     fig, ax = plt.subplots(figsize=(6,5), dpi=300)
#     for i, case in enumerate(CaseList.keys()):
#         ax.plot(strain_values[i], color=colors[i], marker=markers[i], 
#                 markersize=5, linestyle='-',linewidth=2, label=labels[i])
    
#     ax.set_ylabel(label, fontweight='bold', fontsize=14)
#     ax.set_xlabel('Time Step', fontweight='bold', fontsize=14)
#     ax.set_xlim(0, len(CaseList[case]))
#     ax.legend()
    
#     plt.savefig(f'{selected_ly}_{field}_TimeHistory.png', dpi=300, bbox_inches='tight')

#     plt.show()
#     plt.close()
    
  
def plot_EMax(fields, myData, selected_ly, label_selected_ly):
    # Assuming 'colors' is defined with enough colors for your fields
    colors = plt.cm.GnBu(np.linspace(0, 1, 11))
    labels = [r"$\epsilon_{11}$",
             r"$\epsilon_{22}$",
             r"$\epsilon_{33}$",
             r"$\epsilon_{12}$",
             r"$\epsilon_{13}$",
             r"$\epsilon_{23}$"]

    legend_labels = ['6.0 kips, SL=0%', 
                     '6.0 kips, SL=2%', 
                     '4.2 kips, SL=0%', 
                     '4.2 kips, SL=2%', 
                     '9.5 kips, SL=0%', 
                     '4.7 kips, SL=0%', 
                     '4.7 kips, SL=6%', 
                     '7.0 kips, SL=0%', 
                     '7.0 kips, SL=3%', 
                     '4.2 kips, SL=0%', 
                     '4.2 kips, SL=6%']
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    width = 0.075  # Bar width
    num_cases = 11  # Number of cases per field
    num_fields = len(fields)
    
    E_results = []

    for field_idx, field in enumerate(fields):
        
        for case_idx, (case, values) in enumerate(CaseList.items()):
            max_strains = []
            locations = []

            # Define the specific time steps for each case
            if case_idx == 0:  # First case
                time_steps = range(4, 15)
            elif case_idx == 1:  # Second case
                time_steps = range(4, 15)
            elif case_idx == 2:  # Third case
                time_steps = range(4, 15)
            else:
                time_steps = values  # Default to all time steps if case index is out of range
            
            for ts in time_steps:
                data_ts = myData[case_idx][selected_ly][ts-1].copy()
                # Initialize strain_value at a safe default, if necessary
                strain_value = 0  # Default value, in case the field does not match
                
                if field == 'U2':
                    label = r'Deflection $U_2$ (mm)'
                    strain_value = data_ts[field].max()
                    location = data_ts.loc[data_ts[field].idxmax()]
            
                elif field in ['E11', 'E22', 'E33', 'E12', 'E13', 'E23']:
                    if field == 'E11':
                        strain_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                        location = data_ts.loc[data_ts[field].idxmax()]
                        
                    elif field == 'E22':
                        strain_value = abs(data_ts[field].min()) * 1000000  # Absolute value of the minimum
                        location = data_ts.loc[data_ts[field].idxmin()]
                        
                    elif field == 'E33':
                        strain_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                        location = data_ts.loc[data_ts[field].idxmax()]
                        
                    elif field == 'E12':
                        max_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                        min_value = abs(data_ts[field].min()) * 1000000  # Absolute value of the minimum
                        
                        if max_value>min_value:
                            strain_value = max_value
                            location = data_ts.loc[data_ts[field].idxmax()]
                        else:
                            strain_value = min_value
                            location = data_ts.loc[data_ts[field].idxmin()]

                    elif field == 'E13':
                        max_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                        min_value = abs(data_ts[field].min()) * 1000000  # Absolute value of the minimum
                        
                        if max_value>min_value:
                            strain_value = max_value
                            location = data_ts.loc[data_ts[field].idxmax()]
                        else:
                            strain_value = min_value
                            location = data_ts.loc[data_ts[field].idxmin()]
                            
                    elif field == 'E23':
                        max_value = abs(data_ts[field].max()) * 1000000  # Absolute value of the maximum
                        min_value = abs(data_ts[field].min()) * 1000000  # Absolute value of the minimum
                        
                        if max_value>min_value:
                            strain_value = max_value
                            location = data_ts.loc[data_ts[field].idxmax()]
                        else:
                            strain_value = min_value
                            location = data_ts.loc[data_ts[field].idxmin()]

                #output the location of strain_value
                locations.append(location)
                max_strains.append(strain_value)
                
            # Calculate the position for each bar to create a group effect
            positions = [x + (case_idx - (num_cases / 2)) * width for x in range(num_fields)]
            # Plot
            ax.bar(positions[field_idx], max(max_strains), width, label=f'{case}', color=colors[case_idx], edgecolor='black')

            #print max strain and location
            max_strain_index = max_strains.index(max(max_strains))
            max_strain_location = locations[max_strain_index]
            print(f'{case} - {field} - Max Strain: {max(max_strains)} - Location: X {max_strain_location["Xn_elem"]}, Y {max_strain_location["Yn_elem"]}, Z {max_strain_location["Zn_elem"]}')

            E_results.append({
                            'Case': case,
                            'Field': field,
                            'Max Strain': max(max_strains),
                            'X Loc': max_strain_location["Xn_elem"],
                            'Y Loc': max_strain_location["Yn_elem"],
                            'Z Loc': max_strain_location["Zn_elem"]
                           })
    
    df_results = pd.DataFrame(E_results)
    with pd.ExcelWriter(f'FD_{selected_ly}_MaxStrains.xlsx', engine='openpyxl') as writer:
                        df_results.to_excel(writer, index=False, sheet_name=selected_ly)

    ax.set_ylabel(r"Maximum Strain ($\mu\epsilon$)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Strain Fields", fontweight='bold', fontsize=14)
    ax.set_xticks([i-width/2 for i in range(num_fields)])  # Center x-ticks between groups
    ax.set_xticklabels(labels, fontsize=14)  # Use custom labels
    ax.legend(fontsize=10, labels=legend_labels)
    ax.title.set_text(f'{label_selected_ly}')
    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:num_cases], legend_labels[:num_cases], fontsize=10)

    #plt.savefig(f'{selected_ly}_MaxStrains.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def plot_SMax(fields, myData, selected_ly, label_selected_ly):
    # Assuming 'colors' is defined with enough colors for your fields
    colors = plt.cm.GnBu(np.linspace(0, 1, 11))
    labels = [r"$S_{11}$",
              r"$S_{22}$",
              r"$S_{33}$",
              r"$S_{12}$",
              r"$S_{13}$",
              r"$S_{23}$"]

    legend_labels = ['6.0 kips, SL=0%', 
                     '6.0 kips, SL=2%', 
                     '4.2 kips, SL=0%', 
                     '4.2 kips, SL=2%', 
                     '9.5 kips, SL=0%', 
                     '4.7 kips, SL=0%', 
                     '4.7 kips, SL=6%', 
                     '7.0 kips, SL=0%', 
                     '7.0 kips, SL=3%', 
                     '4.2 kips, SL=0%', 
                     '4.2 kips, SL=6%']
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    width = 0.075  # Bar width
    num_cases = 11  # Number of cases per field
    num_fields = len(fields)
    
    S_results = []

    for field_idx, field in enumerate(fields):
        
        for case_idx, (case, values) in enumerate(CaseList.items()):
            max_stresses = []
            locations = []

            # Define the specific time steps for each case
            if case_idx == 0:  # First case
                time_steps = range(4, 15)
            elif case_idx == 1:  # Second case
                time_steps = range(4, 15)
            elif case_idx == 2:  # Third case
                time_steps = range(4, 15)
            else:
                time_steps = values  # Default to all time steps if case index is out of range
            
            for ts in time_steps:
                data_ts = myData[case_idx][selected_ly][ts-1].copy()
                # Initialize strain_value at a safe default, if necessary
                stress_value = 0  # Default value, in case the field does not match
                
                if field in ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']:
                    if field == 'S11':
                        stress_value = abs(data_ts[field].max())   # Absolute value of the maximum
                        location = data_ts.loc[data_ts[field].idxmax()]
                        
                    elif field == 'S22':
                        stress_value = abs(data_ts[field].min())   # Absolute value of the minimum
                        location = data_ts.loc[data_ts[field].idxmin()]
                        
                    elif field == 'S33':
                        stress_value = abs(data_ts[field].max())   # Absolute value of the maximum
                        location = data_ts.loc[data_ts[field].idxmax()]
                        
                    elif field == 'S12':
                        max_value = abs(data_ts[field].max())   # Absolute value of the maximum
                        min_value = abs(data_ts[field].min())   # Absolute value of the minimum
                        
                        if max_value>min_value:
                            stress_value = max_value
                            location = data_ts.loc[data_ts[field].idxmax()]
                        else:
                            stress_value = min_value
                            location = data_ts.loc[data_ts[field].idxmin()]

                    elif field == 'S13':
                        max_value = abs(data_ts[field].max())   # Absolute value of the maximum
                        min_value = abs(data_ts[field].min())   # Absolute value of the minimum
                        
                        if max_value>min_value:
                            stress_value = max_value
                            location = data_ts.loc[data_ts[field].idxmax()]
                        else:
                            stress_value = min_value
                            location = data_ts.loc[data_ts[field].idxmin()]
                            
                    elif field == 'S23':
                        max_value = abs(data_ts[field].max())   # Absolute value of the maximum
                        min_value = abs(data_ts[field].min())   # Absolute value of the minimum
                        
                        if max_value>min_value:
                            stress_value = max_value
                            location = data_ts.loc[data_ts[field].idxmax()]
                        else:
                            stress_value = min_value
                            location = data_ts.loc[data_ts[field].idxmin()]

                #output the location of strain_value
                locations.append(location)
                max_stresses.append(stress_value)
                
            # Calculate the position for each bar to create a group effect
            positions = [x + (case_idx - (num_cases / 2)) * width for x in range(num_fields)]
            # Plot
            ax.bar(positions[field_idx], max(max_stresses), width, label=f'{case}', color=colors[case_idx], edgecolor='black')

            #print max strain and location
            max_stress_index = max_stresses.index(max(max_stresses))
            max_stress_location = locations[max_stress_index]
            print(f'{case} - {field} - Max Stress: {max(max_stresses)} - Location: X {max_stress_location["Xn_elem"]}, Y {max_stress_location["Yn_elem"]}, Z {max_stress_location["Zn_elem"]}')

            S_results.append({
                            'Case': case,
                            'Field': field,
                            'Max Stress': max(max_stresses),
                            'X Loc': max_stress_location["Xn_elem"],
                            'Y Loc': max_stress_location["Yn_elem"],
                            'Z Loc': max_stress_location["Zn_elem"]
                           })           

    df_results = pd.DataFrame(S_results)
    with pd.ExcelWriter(f'FD_{selected_ly}_MaxStress.xlsx', engine='openpyxl') as writer:
                        df_results.to_excel(writer, index=False, sheet_name=selected_ly)

    ax.set_ylabel(r"Maximum Stress ($\sigma$)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Stress Fields", fontweight='bold', fontsize=14)
    ax.set_xticks([i-width/2 for i in range(num_fields)])  # Center x-ticks between groups
    ax.set_xticklabels(labels, fontsize=14)  # Use custom labels
    ax.legend(fontsize=10, labels=legend_labels)
    ax.title.set_text(f'{label_selected_ly}')
    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:num_cases], legend_labels[:num_cases], fontsize=10)

    #plt.savefig(f'{selected_ly}_MaxStresses.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


##################################################
########             PLOTS                ######## 
##################################################

# plot_ETH('U2', myData_layers)
# plot_ETH('E22', myData_layers)
# plot_ETH('E11', myData_layers)
# plot_ETH('E33', myData_layers)
# plot_ETH('E23', myData_layers)


# Define fields to plot
plot_EMax(['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], myData_layers, 'Surf', 'NearSurf') 
plot_EMax(['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], myData_layers, 'AC1', 'Surface Layer') 
plot_EMax(['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], myData_layers, 'AC2', 'Intermediate Layer') 
plot_EMax(['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], myData_layers, 'AC3', 'Binder Layer') 
plot_EMax(['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], myData_layers, 'B1', 'Base') 
plot_EMax(['E11', 'E22', 'E33', 'E12', 'E13', 'E23'], myData_layers, 'SG1', 'Subgrade') 



# plot_SMax(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], myData_layers, 'Surf', 'NearSurf') 
# plot_SMax(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], myData_layers, 'AC1', 'Surface Layer') 
# plot_SMax(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], myData_layers, 'AC2', 'Intermediate Layer') 
# plot_SMax(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], myData_layers, 'AC3', 'Binder Layer') 
# plot_SMax(['S11', 'S22', 'S33', 'S12', 'S13', 'S23'], myData_layers, 'SG1', 'Subgrade') 
# %%
