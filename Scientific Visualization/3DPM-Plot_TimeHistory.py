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

CaseList = ['CC71DS_P5_AC1S_B1_SB1_SG1']
tstep = [9, 16]                          # Range of time steps to be analyzed

# Model Dimensions (Update per case)
L = 32677.0        # Length of the Model
Xw = 1677.0        # Length of the Wheel Path
B = 32750.0        # Width of the Model
Depth = 15000.0    # Total Depth[mm] of the Model

Structure = ['AC1', 'B1', 'SB1', 'SG1']   # Pavement Layers
MyLabels = ['P-401', 'P-209', 'P-154', 'Subgrade'] # Plot Labels
Thicks = [75.0, 150.0, 500.0, 14275.0]    # Thickness of each layer

user = 'johannc2'
directory = f'C:/Users/{user}/Box/FAA Data Project/04_FEM/00_FEM DATA/FAA_South/FAA_South_Responses/{CaseList[0]}/'

# Layer of Analysis
la = 'AC1'

##################################################
########     Preliminary Calculations     ######## 
##################################################

myData = []
myData_all = []
myE11max, myE22max, myE23max, myE33max = [], [], [], []
myE11min, myE22min, myE23min, myE33min = [], [], [], []
myS11max, myS22max, myS23max, myS33max = [], [], [], []
myS11min, myS22min, myS23min, myS33min = [], [], [], []

y_ranges = {}
cumulative_thickness = 0
for layer, thickness in zip(Structure, Thicks):
    y_ranges[layer] = (Depth - cumulative_thickness - thickness, Depth - cumulative_thickness)
    cumulative_thickness += thickness

for c in CaseList:
    for ts in range(tstep[0], tstep[-1]):
        filename = f'{c}_3DResponse_tire{ts}.txt'
        filepath = os.path.join(directory, filename)
        
        # Remove the header
        df = pd.read_csv(filepath, sep='\t')
        data_all = df.copy()
        data_all = data_all.sort_values(by='Node', ascending=True)
        
        y_lower, y_upper = y_ranges[la]
        
        # Sort the dataframe by 'Node'. Lowes node numbers first.
        df = df.sort_values(by='Node', ascending=True)
        # Filter dataframe based on the Yn coordinate
        df_layer = df[df['Yn_elem'].between(y_lower, y_upper)]
        
        if la == Structure[0]:  # First layer (top)
            df_new = df_layer.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='first')
            
        elif la == Structure[-1]:  # Last layer (bottom)
            df_new = df_layer.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
        
        else:  # Intermediate layers
    
            df_int_high = df_layer[df_layer['Yn_elem'] == y_upper]
            df_int_high = df_int_high.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
        
            df_int_low = df_layer[df_layer['Yn_elem'] == y_lower]
            df_int_low = df_int_low.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='first')
        
            df_new = pd.concat([
                df_layer[(df_layer['Yn_elem'] > y_lower) & (df_layer['Yn_elem'] < y_upper)],
                df_int_low,
                df_int_high
            ])        
        
        dataly = df_new.copy()
        
        ##################################################
        ########        STRAIN EXTRACTION         ######## 
        ##################################################
        
        myE11max.append(dataly['E11'].max())
        myE11min.append(dataly['E11'].min())
        
        myE22max.append(dataly['E22'].max())
        myE22min.append(dataly['E22'].min())
        
        myE23max.append(dataly['E23'].max())
        myE23min.append(dataly['E23'].min())
        
        myE33max.append(dataly['E33'].max())
        myE33min.append(dataly['E33'].min())
    
        ##################################################
        ########        STRESS EXTRACTION         ######## 
        ##################################################
                
        myS11max.append(dataly['S11'].max())
        myS11min.append(dataly['S11'].min())
        
        myS22max.append(dataly['S22'].max())
        myS22min.append(dataly['S22'].min())
        
        myS23max.append(dataly['S23'].max())
        myS23min.append(dataly['S23'].min())
        
        myS33max.append(dataly['S33'].max())
        myS33min.append(dataly['S33'].min())

        myData_all.append(data_all)
        myData.append(dataly)


# Get the index of the maximum value within E11max
max_E11_idx = myE11max.index(max(myE11max))
min_E11_idx = myE11min.index(min(myE11min))

max_E22_idx = myE22max.index(max(myE22max))
min_E22_idx = myE22min.index(min(myE22min))

max_E23_idx = myE23max.index(max(myE23max))
min_E23_idx = myE23min.index(min(myE23min))

max_E33_idx = myE33max.index(max(myE33max))
min_E33_idx = myE33min.index(min(myE33min))


max_S11_idx = myS11max.index(max(myS11max))
min_S11_idx = myS11min.index(min(myS11min))

max_S22_idx = myS22max.index(max(myS22max))
min_S22_idx = myS22min.index(min(myS22min))

max_S23_idx = myS23max.index(max(myS23max))
min_S23_idx = myS23min.index(min(myS23min))

max_S33_idx = myS33max.index(max(myS33max))
min_S33_idx = myS33min.index(min(myS33min))


##################################################
########       LAYER SEGMENTATION         ######## 
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
########           STRAIN PLOTS           ######## 
##################################################

# Longitudinal Profile Along the Z Axis
def plot_E_1DY(strain_component, myData):
    
    if strain_component == 'E11':
        data = myData[max_E11_idx].copy()
        label = 'Longitudinal Strain'
        idx = max_E11_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        #location
        strain_filtered = [myData[i].loc[(myData[i]['Xn_elem'] == location['Xn_elem']) & (myData[i]['Yn_elem'] == location['Yn_elem']) & (myData[i]['Zn_elem'] == location['Zn_elem']), strain_component] for i in range(len(myData))]
        strain_values = [series.iloc[0]*1000000 for series in strain_filtered]
         
    if strain_component == 'E33':
        data = myData[max_E33_idx].copy()
        label = 'Transverse Strain'
        idx = max_E33_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        strain_filtered = [myData[i].loc[(myData[i]['Xn_elem'] == location['Xn_elem']) & (myData[i]['Yn_elem'] == location['Yn_elem']) & (myData[i]['Zn_elem'] == location['Zn_elem']), strain_component] for i in range(len(myData))]
        strain_values = [series.iloc[0]*1000000 for series in strain_filtered]
        
    if strain_component == 'E22':
        data = myData[min_E22_idx].copy()
        label = 'Vertical Strain'
        idx = min_E22_idx
        max_strain_value = data[strain_component].min()
        location = data.loc[data[strain_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        strain_filtered = [myData[i].loc[(myData[i]['Xn_elem'] == location['Xn_elem']) & (myData[i]['Yn_elem'] == location['Yn_elem']) & (myData[i]['Zn_elem'] == location['Zn_elem']), strain_component] for i in range(len(myData))]
        strain_values = [series.iloc[0]*1000000 for series in strain_filtered]

    if strain_component == 'E23':
        data = myData[min_E23_idx].copy()
        label = 'Shear Strain'
        
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = max_E23_idx
            max_strain_value = data[strain_component].max()
            location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
            strain_filtered = [myData[i].loc[(myData[i]['Xn_elem'] == location['Xn_elem']) & (myData[i]['Yn_elem'] == location['Yn_elem']) & (myData[i]['Zn_elem'] == location['Zn_elem']), strain_component] for i in range(len(myData))]
            strain_values = [series.iloc[0]*1000000 for series in strain_filtered]
        else:
            idx = min_E23_idx
            max_strain_value = data[strain_component].min()
            location = data.loc[data[strain_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
            strain_filtered = [myData[i].loc[(myData[i]['Xn_elem'] == location['Xn_elem']) & (myData[i]['Yn_elem'] == location['Yn_elem']) & (myData[i]['Zn_elem'] == location['Zn_elem']), strain_component] for i in range(len(myData))]
            strain_values = [series.iloc[0]*1000000 for series in strain_filtered]
        
    time_labels = [i for i in range(tstep[0], tstep[-1])]

    # Create the line plot
    plt.figure(figsize=(8, 5))
    plt.plot(strain_values, color='b', marker='o', linewidth=2.0, markersize=6.0)
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    plt.xticks(np.arange(0, len(time_labels), 1), time_labels)
        
    plt.ylabel(f'{label}'+ ' ' + f'{strain_component} ($\mu\epsilon$)', fontweight='bold', fontsize=12)
    plt.xlabel('Time Step', fontweight='bold', fontsize=12)
    
    # Plot point at location of the maximum strain value
    plt.plot(idx, max_strain_value*1000000, color='r', 
             marker='o', markersize=6.0, markeredgecolor='black', markeredgewidth=1.0)
    
    # Textbox
    # textstr = (f"${strain_component}_{{max}}$: {max_strain_value*1000000:.1f} $\mu\epsilon$")
    # # Position the text box in figure coords to ensure it's always in the same position regardless of data
    # props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    # plt.text(idx, max_strain_value*1000000 + 25, textstr, fontsize=12, verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_Time_History.png', dpi=500, bbox_inches='tight')
    plt.show()
    
      
##################################################
########             PLOTS                ######## 
##################################################

plot_E_1DY('E22', myData)
plot_E_1DY('E11', myData)
plot_E_1DY('E33', myData)
plot_E_1DY('E23', myData)
