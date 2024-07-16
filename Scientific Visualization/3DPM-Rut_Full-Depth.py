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
#              Vertical Profile (Depht) Plotting 
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

CaseList = ['FD_P1_SL0']
tstep = [8, 12]           # Range of time steps to be analyzed

# Model Dimensions (Update per case)
L = 9520.0        # Length of the Model
Xw = 1320.0        # Length of the Wheel Path
B = 8432.0        # Width of the Model
Depth = 5000.0    # Total Depth[mm] of the Model

Structure = ['AC1', 'AC2', 'AC3', 'B1', 'SG1']   # Pavement Layers
MyLabels = ['WS', 'IM', 'BL', 'Base', 'Subgrade'] # Plot Labels
Thicks = [50.0, 55.0, 145.0, 305.0, 4445.0]    # Thickness of each layer
NElem = [8, 8, 12, 12, 20]

user = 'johan'
directory = f'C:/Users/{user}/Box/R27-252 EV/Tasks/Task 3 - Pavement FEM/Post-Processing/{CaseList[0]}/'

# Layer of Analysis
la = 'AC3'

##################################################
########     Preliminary Calculations     ######## 
##################################################

myData = []
myData_all = []
myE11max, myE22max, myE23max, myE33max = [], [], [], []
myE11min, myE22min, myE23min, myE33min = [], [], [], []

y_ranges = {}
cumulative_thickness = 0
for layer, thickness in zip(Structure, Thicks):
    y_ranges[layer] = (Depth - cumulative_thickness - thickness, Depth - cumulative_thickness)
    cumulative_thickness += thickness


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
            dataframes[i]['Layer'] = lay
            
        elif i == len(Structure) - 1:  # Last layer (bottom)
            dataframes[i] = dfl.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
            dataframes[i]['Layer'] = lay
            
        else:  # Intermediate layers
            df_int_high = dfl[dfl['Yn_elem'] == y_upper].drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
            df_int_low = dfl[dfl['Yn_elem'] == y_lower].drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='first')
            dataframes[i] = pd.concat([dfl[(dfl['Yn_elem'] > y_lower) & (dfl['Yn_elem'] < y_upper)], df_int_low, df_int_high])
            dataframes[i]['Layer'] = lay
            
    return dataframes


##################################################
########          PROCESSING              ######## 
##################################################


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
            df_new['Layer'] = Structure[0]
            
        elif la == Structure[-1]:  # Last layer (bottom)
            df_new = df_layer.drop_duplicates(subset=['Xn_elem', 'Yn_elem', 'Zn_elem'], keep='last')
            df_new['Layer'] = Structure[-1]
        
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
               
            df_new['Layer'] = la 
        
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

        myData_all.append(data_all)
        myData.append(dataly)


# Get the index of the maximum value 
max_E11_idx = myE11max.index(max(myE11max))
min_E11_idx = myE11min.index(min(myE11min))

max_E22_idx = myE22max.index(max(myE22max))
min_E22_idx = myE22min.index(min(myE22min))

max_E23_idx = myE23max.index(max(myE23max))
min_E23_idx = myE23min.index(min(myE23min))

max_E33_idx = myE33max.index(max(myE33max))
min_E33_idx = myE33min.index(min(myE33min))

        
##################################################
########           STRAIN PLOTS           ######## 
##################################################

# Longitudinal Profile Along the Z Axis
def plot_E_1DY(strain_component, dat):
    
    data = dat.copy()

    if strain_component == 'E11':
        label = 'Longitudinal Strain'
        idx = max_E11_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if strain_component == 'E33':
        label = 'Transverse Strain'
        idx = max_E33_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if strain_component == 'E22':
        label = 'Vertical Strain'
        idx = min_E22_idx
        max_strain_value = data[strain_component].min()
        location = data.loc[data[strain_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if strain_component == 'E23':
        label = 'Shear Strain'
        
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = max_E23_idx
            max_strain_value = data[strain_component].max()
            location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = min_E23_idx
            max_strain_value = data[strain_component].min()
            location = data.loc[data[strain_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    data_all = myData_all[idx].copy()
    
    # Filter the data for the Yn and Zn coordinates that correspond to the maximum strain value
    filtered_data = data_all[(data_all['Xn_elem'] == location['Xn_elem']) & (data_all['Zn_elem'] == location['Zn_elem'])]
    # Sort the filtered data based on the Yn coordinate, and based on Node label
    filtered_data_sorted = filtered_data.sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
    
    # Output filtered_data_sorted to a CSV file

    filtered_data_sorted.insert(0, 'Layer', 'None')
    # Label the first 9 rows in the Layer column as 'AC1'
    cumsum = [0] + list(pd.Series(NElem).cumsum())
    filtered_data_sorted.loc[filtered_data_sorted.index[cumsum[0] : cumsum[1]+1], 'Layer'] = 'AC1'
    filtered_data_sorted.loc[filtered_data_sorted.index[cumsum[1]+1 : cumsum[2]+2], 'Layer'] = 'AC2'
    filtered_data_sorted.loc[filtered_data_sorted.index[cumsum[2]+2 : cumsum[3]+3], 'Layer'] = 'AC3'
    filtered_data_sorted.loc[filtered_data_sorted.index[cumsum[3]+3 : cumsum[4]+4], 'Layer'] = 'B1'
    filtered_data_sorted.loc[filtered_data_sorted.index[cumsum[4]+4 :], 'Layer'] = 'SG1'
        
    filtered_data_sorted.to_csv('Cumulative_E22_' + CaseList[0] + '.csv', index=False)
    
    # Create the line plot
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(filtered_data_sorted[strain_component]*1000000, filtered_data_sorted['Yn_elem'], 
             color='b', marker='o', linewidth=4.0, markersize=2.5)
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
        
    #plt.title(f'Vertical Profile of {strain_component}', fontweight='bold', fontsize=14)
    plt.xlabel(f'{label}'+ ' ' + f'{strain_component} ($\mu\epsilon$)', fontweight='bold', fontsize=14)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Plot point at location of the maximum strain value
    plt.plot(max_strain_value*1000000, location['Yn_elem'], color='r', 
             marker='d', markersize=8.0, markeredgecolor='black', markeredgewidth=1.0)
    plt.ylim(ymin=filtered_data_sorted['Yn_elem'].min(), ymax=Depth)
    
    # Point coordinates for textbox
    ax = plt.gca()
    fig = plt.gcf()
    data_coord = ax.transData.transform((max_strain_value*1000000, location['Yn_elem']))
    fig_coord = fig.transFigure.inverted().transform(data_coord)
    textbox_y = fig_coord[1] - 0.065
    textbox_x = fig_coord[0] + 0.075
    
    # Draw a horizontal plot at a given Yn coordinate
    for t in range(len(Thicks)):
        plt.axhline(y=y_ranges[Structure[t]][1], color='k', linestyle='-', linewidth=0.75)
        #add a label in the right corner of the horizontal line
        
        ax_height = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
        y_axes_coord = (y_ranges[Structure[t]][1] - plt.gca().get_ylim()[0]) / ax_height
        plt.text(0.026, y_axes_coord - 0.02, MyLabels[t], transform=plt.gca().transAxes, fontsize=12,
             horizontalalignment='left', verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.0, edgecolor='black', boxstyle='round,pad=0.25'))
        
    plt.ylim(ymin=filtered_data_sorted['Yn_elem'].min(), ymax=Depth)

    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
        
    textstr = (f"${strain_component}_{{{la}max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$")
    
    # Position the text box in figure coords to ensure it's always in the same position regardless of data
    props = dict(boxstyle="round4,pad=0.30", edgecolor='grey', facecolor=(1, 1, 1, 1), linewidth=2)
    plt.gcf().text(textbox_x, textbox_y, textstr, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    
    plt.grid(True)
    
    #plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_1DY.png', dpi=500, bbox_inches='tight')
    plt.show()
    


def plot_E_1DY_multi(strain_components, dat):
    # Initialize plot
    plt.figure(figsize=(5, 5), dpi=300)
    
    textstr = ['' for _ in strain_components]
    
    for strain_component in strain_components:
        
        #get index of strain component
        if strain_component == 'E11':
            idx_case=0
        elif strain_component == 'E33':
            idx_case=1
        
        data = dat[idx_case].copy()
        
        if strain_component == 'E11':
            color = 'blue'
            label = f'$E_{{11}}$'
            idx = max_E11_idx
        elif strain_component == 'E33':
            color = 'green'
            label = f'$E_{{33}}$'
            idx = max_E33_idx
        else:
            continue  # Skip if the component is not E11 or E33
        
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
        data_all = myData_all[idx].copy()
        filtered_data = data_all[(data_all['Xn_elem'] == location['Xn_elem']) & (data_all['Zn_elem'] == location['Zn_elem'])]
        filtered_data_sorted = filtered_data.sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
        
        # Plotting for each strain component
        plt.plot(filtered_data_sorted[strain_component]*1000000, filtered_data_sorted['Yn_elem'], 
                 color=color, marker='o', linewidth=4.0, markersize=2, label=label)
        
        # Mark maximum strain value
        plt.plot(max_strain_value*1000000, location['Yn_elem'], color=color, 
                 marker='d', markersize=8.0, markeredgecolor='black', markeredgewidth=1.0)
        
        textstr[idx_case] = (f"${strain_component}_{{{la}max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$")
        
        
    for t in range(len(Thicks)):
        plt.axhline(y=y_ranges[Structure[t]][1], color='k', linestyle='-', linewidth=0.75)
        #add a label in the right corner of the horizontal line
        
        ax_height = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
        y_axes_coord = (y_ranges[Structure[t]][1] - plt.gca().get_ylim()[0]) / ax_height
        
        if t==0:
            plt.text(0.026, y_axes_coord + 0.02, MyLabels[t], transform=plt.gca().transAxes, fontsize=12,
             horizontalalignment='left', verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.0, edgecolor='black', boxstyle='round,pad=0.25'))
        else:
            plt.text(0.026, y_axes_coord - 0.02, MyLabels[t], transform=plt.gca().transAxes, fontsize=12,
             horizontalalignment='left', verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.0, edgecolor='black', boxstyle='round,pad=0.25'))
        
    plt.ylim(ymin=filtered_data_sorted['Yn_elem'].min(), ymax=Depth)

    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))  
        
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    plt.xlabel('Tensile Strain ($\mu\epsilon$)', fontweight='bold', fontsize=14)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
    plt.ylim(ymin=filtered_data_sorted['Yn_elem'].min(), ymax=Depth)
    

    props = dict(boxstyle="round4,pad=0.30", edgecolor='grey', facecolor=(1, 1, 1, 1), linewidth=2)
    plt.gcf().text(0.7, 0.2, textstr[0], transform=plt.gca().transAxes, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    plt.gcf().text(0.7, 0.1, textstr[1], transform=plt.gca().transAxes, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    
    #plt.savefig(f'{la}'+ '_E11_E33_1DY.png', dpi=500, bbox_inches='tight')
    plt.show()
    
       
    
##################################################
########             PLOTS                ######## 
##################################################

plot_E_1DY('E22', myData[min_E22_idx])
plot_E_1DY('E11', myData[max_E11_idx])
plot_E_1DY('E33', myData[max_E33_idx])
plot_E_1DY('E23', myData[min_E23_idx])

plot_E_1DY_multi(['E11', 'E33'], [myData[max_E11_idx], myData[max_E33_idx]])   
