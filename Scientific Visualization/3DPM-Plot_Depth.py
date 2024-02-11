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

CaseList = ['CC71DS_P5_AC1S_B1_SB1_SG1']
tstep = [8, 13]                          # Range of time steps to be analyzed

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
la = 'B1'

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
    
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_1DY.png', dpi=500, bbox_inches='tight')
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
    
    plt.savefig(f'{la}'+ '_E11_E33_1DY.png', dpi=500, bbox_inches='tight')
    plt.show()
    
      
def plot_EZY_2D(strain_component, dat):

    data = dat.copy()
    data['Zn_elem'] = data['Zn_elem'] - (B/2)
    
    if strain_component == 'E11':
        label = 'Longitudinal Strain'
        idx = max_E11_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if strain_component == 'E33':
        label = 'Transverse Strain'
        idx = max_E33_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[data[strain_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if strain_component == 'E22':
        label = 'Vertical Strain'
        idx = min_E22_idx
        max_strain_value = data[strain_component].min()
        location = data.loc[data[strain_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
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
        legend_height = 0.50
    
    dataframes = process_layers(myData_all[idx].copy())
    num_layers = len(dataframes)
    
    
    yy_ranges=y_ranges.copy()
    yy_ranges['SG1'] = (yy_ranges['SG1'][1]-500, yy_ranges['SG1'][1])
    
    total_y_range = sum(yy_ranges[la][1] - yy_ranges[la][0] for la in Structure)
    relative_heights = [(yy_ranges[la][1] - yy_ranges[la][0]) / total_y_range for la in Structure]

    fig, axs = plt.subplots(num_layers, 1, figsize=(8, 7), gridspec_kw={'height_ratios': relative_heights}, sharex=True)

    plt.subplots_adjust(hspace=0.1)
    
    # Global Parameters
    global_min = min([df[strain_component].min() for df in dataframes])
    global_max = max([df[strain_component].max() for df in dataframes])
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    for k, dfs in enumerate(dataframes):
        ax = axs[k] if num_layers>1 else axs
        
        dfs['Zn_elem'] = dfs['Zn_elem'] - (B/2) 
        data_at_max_x_raw = dfs [dfs ['Xn_elem'] == location['Xn_elem']]
        data_at_max_x = data_at_max_x_raw .sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
        
        # Create a grid to interpolate onto
        Z_unique = np.linspace(data_at_max_x['Zn_elem'].min(), data_at_max_x['Zn_elem'].max(), 1000)
        Y_unique = np.linspace(data_at_max_x['Yn_elem'].min(), data_at_max_x['Yn_elem'].max(), 1000)
        Z_grid, Y_grid = np.meshgrid(Z_unique, Y_unique)
    
        # Interpolate the data onto this grid
        points = data_at_max_x[['Zn_elem', 'Yn_elem']].values
        values = data_at_max_x[strain_component].values
        grid_z0 = griddata(points, values, (Z_grid, Y_grid), method='cubic')
    
        # Create the contour plot
        contour = ax.contourf(Z_grid, Y_grid, grid_z0 * 1000000, cmap='magma', levels=100, vmin=global_min * 1000000, vmax=global_max * 1000000)
        ax.grid(True, linestyle='--', color='0.8', linewidth=0.5)
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
        
        # Add a textbox with layer name
        ax.text(0.075, 0.5, MyLabels[k], transform=ax.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.25'))
        
        if k == num_layers - 1:  # Only set xlabel on the last subplot
            ax.set_xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=16)


        ax.yaxis.set_tick_params(labelsize=14)
        
    fig.text(0.025, 0.5, 'Depth (mm)', va='center', rotation='vertical', fontweight='bold', fontsize=16)
    
    # Add a color bar
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=global_min * 1000000, vmax=global_max * 1000000)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label(f'{label}'+ ' ' + f'{strain_component}' + ' ' + '$(\mu\epsilon)$', fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    
    # plt.title(f'Contour Plot of {strain_component} on ZY Plane', fontweight='bold', fontsize=14)
    plt.xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=14)

    # # Construct the textbox string with newlines for each piece of information
    # textstr = (f"${strain_component}_{{{la}max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$")
    
    # # Position the text box in figure coords
    # props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    # fig.text(0.95, legend_height, textstr, transform=axs[Structure.index(la)].transAxes, fontsize=12, bbox=props, 
    #         horizontalalignment='right', verticalalignment='bottom')
    
    plt.grid(True)
    # save figure
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_ZY-Section.png', dpi=500, bbox_inches='tight')
    plt.show()
        

# Longitudinal Cut Along the XY Plane
def plot_EXY_2D(strain_component, dat):

    data = dat.copy()
    data['Xn_elem'] = data['Xn_elem'] - (L/2 - Xw/2)
    
    if strain_component == 'E11':
        label = 'Longitudinal Strain'
        idx = max_E11_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if strain_component == 'E33':
        label = 'Transverse Strain'
        idx = max_E33_idx
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if strain_component == 'E22':
        label = 'Vertical Strain'
        idx = min_E22_idx
        max_strain_value = data[strain_component].min()
        location = data.loc[min_E22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if strain_component == 'E23':
        label = 'Shear Strain'
        
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = max_E23_idx
            max_strain_value = data[strain_component].max()
            location = data.loc[max_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = min_E23_idx
            max_strain_value = data[strain_component].min()
            location = data.loc[min_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
    
    dataframes = process_layers(myData_all[idx].copy())
    num_layers = len(dataframes)
    
    yy_ranges=y_ranges.copy()
    yy_ranges['SG1'] = (yy_ranges['SG1'][1]-500, yy_ranges['SG1'][1])            
                     
    total_y_range = sum(yy_ranges[la][1] - yy_ranges[la][0] for la in Structure)          
    relative_heights = [(yy_ranges[la][1] - yy_ranges[la][0]) / total_y_range for la in Structure]    
    
    fig, axs = plt.subplots(num_layers, 1, figsize=(8, 7), gridspec_kw={'height_ratios': relative_heights}, sharex=True)
    
    plt.subplots_adjust(hspace=0.1)
    
    # Global Parameters
    global_min = min([df[strain_component].min() for df in dataframes])
    global_max = max([df[strain_component].max() for df in dataframes])
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    
    for k, dfs in enumerate(dataframes):
        ax = axs[k] if num_layers>1 else axs
        
        dfs['Xn_elem'] = dfs['Xn_elem'] - (L/2 - Xw/2)
        
        data_at_max_z_raw = dfs[dfs['Zn_elem'] == location['Zn_elem']]
        data_at_max_z = data_at_max_z_raw .sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
        
        # Create a grid to interpolate onto
        X_unique = np.linspace(data_at_max_z['Xn_elem'].min(), data_at_max_z['Xn_elem'].max(), 1000)
        Y_unique = np.linspace(data_at_max_z['Yn_elem'].min(), data_at_max_z['Yn_elem'].max(), 1000)
        X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
        
        # Interpolate the data onto this grid
        points = data_at_max_z[['Xn_elem', 'Yn_elem']].values
        values = data_at_max_z[strain_component].values
        grid_x0 = griddata(points, values, (X_grid, Y_grid), method='cubic')
        
        # Create the contour plot
        contour = ax.contourf(X_grid, Y_grid, grid_x0 * 1000000, cmap='magma', levels=100, vmin=global_min * 1000000, vmax=global_max * 1000000)
        ax.grid(True, linestyle='--', color='0.8', linewidth=0.5)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
        
        # Add a textbox with layer name
        ax.text(0.075, 0.5, MyLabels[k], transform=ax.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.25'))

        if k == num_layers - 1:  # Only set xlabel on the last subplot
            ax.set_xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=16)
            
        ax.yaxis.set_tick_params(labelsize=14)

    fig.text(0.025, 0.5, 'Depth (mm)', va='center', rotation='vertical', fontweight='bold', fontsize=16)    
    
    # Add a color bar
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=global_min * 1000000, vmax=global_max * 1000000)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label(f'{label}'+ ' ' + f'{strain_component}' + ' ' + '$(\mu\epsilon)$', fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    #plt.title(f'Contour Plot of {strain_component} on XY Plane', fontweight='bold', fontsize=14)
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=14)
    
    # Construct the textbox string with newlines for each piece of information
    # textstr = (f"${strain_component}_{{{la}max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$")
    
    # # Position the text box in figure coords
    # props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    # fig.text(0.95, legend_height, textstr, transform=axs[Structure.index(la)].transAxes, fontsize=10, bbox=props, 
    #          horizontalalignment='right', verticalalignment='bottom')
    
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_XY-Section.png', dpi=500, bbox_inches='tight')
    plt.show()
    

##################################################
########           STRESS PLOTS           ######## 
##################################################   
    
def plot_S_1DY(stress_component, dat):
    
    data = dat.copy()

    if stress_component == 'S11':
        label = 'Longitudinal Stress'
        idx = max_S11_idx
        max_stress_value = data[stress_component].max()
        location = data.loc[data[stress_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S33':
        label = 'Transverse Stress'
        idx = max_S33_idx
        max_stress_value = data[stress_component].max()
        location = data.loc[data[stress_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S22':
        label = 'Vertical Stress'
        idx = min_S22_idx
        max_stress_value = data[stress_component].min()
        location = data.loc[data[stress_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S23':
        label = 'Shear Stress'
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = max_S23_idx
            max_stress_value = data[stress_component].max()
            location = data.loc[data[stress_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = min_S23_idx
            max_stress_value = data[stress_component].min()
            location = data.loc[data[stress_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    data_all = myData_all[idx].copy()
    
    # Filter the data for the Yn and Zn coordinates that correspond to the maximum strain value
    filtered_data = data_all[(data_all['Xn_elem'] == location['Xn_elem']) & (data_all['Zn_elem'] == location['Zn_elem'])]
    # Sort the filtered data based on the Yn coordinate, and based on Node label
    filtered_data_sorted = filtered_data.sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
        
    # Create the line plot
    plt.figure(figsize=(6, 6))
    plt.plot(filtered_data_sorted[stress_component], filtered_data_sorted['Yn_elem'], 
             color='b', marker='o', linewidth=3.5, markersize=2.0)
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
        
    #plt.title(f'Vertical Profile of {stress_component}', fontweight='bold', fontsize=14)
    plt.xlabel(f'{label}'+ ' ' + f'{stress_component} (MPa)', fontweight='bold', fontsize=12)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=12)
    
    #set xmin as 1.25 times the minimum stress value
    xmin = 1.50 * filtered_data_sorted[stress_component].min()
    plt.xlim(xmin=xmin)
    
    # Plot point at location of the maximum strain value
    plt.plot(max_stress_value, location['Yn_elem'], color='r', 
             marker='d', markersize=8.0, markeredgecolor='black', markeredgewidth=1.0)
    plt.ylim(ymin=filtered_data_sorted['Yn_elem'].min(), ymax=Depth)
    
    # Point coordinates for textbox
    ax = plt.gca()
    fig = plt.gcf()
    data_coord = ax.transData.transform((max_stress_value, location['Yn_elem']))
    fig_coord = fig.transFigure.inverted().transform(data_coord)
    textbox_y = fig_coord[1] - 0.055
    textbox_x = fig_coord[0] + 0.020
    
    # Draw a horizontal plot at a given Yn coordinate
    for t in range(len(Thicks)):
        plt.axhline(y=y_ranges[Structure[t]][1], color='k', linestyle='-', linewidth=0.75)
        #add a label in the right corner of the horizontal line
        
        ax_height = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
        y_axes_coord = (y_ranges[Structure[t]][1] - plt.gca().get_ylim()[0]) / ax_height
        plt.text(0.026, y_axes_coord - 0.02, MyLabels[t], transform=plt.gca().transAxes, fontsize=10,
             horizontalalignment='left', verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.0, edgecolor='black', boxstyle='round,pad=0.25'))
        
    plt.ylim(ymin=filtered_data_sorted['Yn_elem'].min(), ymax=Depth)

    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
        
    textstr = (f"${stress_component}_{{{la}max}}$: {max_stress_value*1:.2f} MPa")
    
    # Position the text box in figure coords to ensure it's always in the same position regardless of data
    props = dict(boxstyle="round4,pad=0.30", edgecolor='grey', facecolor=(1, 1, 1, 1), linewidth=2)
    plt.gcf().text(textbox_x, textbox_y, textstr, fontsize=10, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{stress_component}_tire{tstep[0]+idx}_VerticalProfile.png', dpi=500, bbox_inches='tight')
    plt.show()


def plot_SZY_2D(stress_component, dat):

    data = dat.copy()
    data['Zn_elem'] = data['Zn_elem'] - (B/2)
    
    if stress_component == 'S11':
        label = 'Longitudinal Stress'
        idx = max_S11_idx
        max_stress_value = data[stress_component].max()
        location = data.loc[data[stress_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if stress_component == 'S33':
        label = 'Transverse Stress'
        idx = max_S33_idx
        max_stress_value = data[stress_component].max()
        location = data.loc[data[stress_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if stress_component == 'S22':
        label = 'Vertical Stress'
        idx = min_S22_idx
        max_stress_value = data[stress_component].min()
        location = data.loc[data[stress_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if stress_component == 'S23':
        label = 'Shear Stress'
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = max_S23_idx
            max_stress_value = data[stress_component].max()
            location = data.loc[data[stress_component].idxmax(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = min_S23_idx
            max_stress_value = data[stress_component].min()
            location = data.loc[data[stress_component].idxmin(), ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
    
    dataframes = process_layers(myData_all[idx].copy())
    num_layers = len(dataframes)
    
    yy_ranges=y_ranges.copy()
    yy_ranges['SG1'] = (yy_ranges['SG1'][1]-500, yy_ranges['SG1'][1])
    
    total_y_range = sum(yy_ranges[la][1] - yy_ranges[la][0] for la in Structure)
    relative_heights = [(yy_ranges[la][1] - yy_ranges[la][0]) / total_y_range for la in Structure]

    fig, axs = plt.subplots(num_layers, 1, figsize=(8, 6), gridspec_kw={'height_ratios': relative_heights}, sharex=True)

    plt.subplots_adjust(hspace=0.1)
    
    # Global Parameters
    global_min = min([df[stress_component].min() for df in dataframes])
    global_max = max([df[stress_component].max() for df in dataframes])
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    for k, dfs in enumerate(dataframes):
        ax = axs[k] if num_layers>1 else axs
        
        dfs['Zn_elem'] = dfs['Zn_elem'] - (B/2) 
        data_at_max_x_raw = dfs [dfs ['Xn_elem'] == location['Xn_elem']]
        data_at_max_x = data_at_max_x_raw .sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
        
        # Create a grid to interpolate onto
        Z_unique = np.linspace(data_at_max_x['Zn_elem'].min(), data_at_max_x['Zn_elem'].max(), 1000)
        Y_unique = np.linspace(data_at_max_x['Yn_elem'].min(), data_at_max_x['Yn_elem'].max(), 1000)
        Z_grid, Y_grid = np.meshgrid(Z_unique, Y_unique)
    
        # Interpolate the data onto this grid
        points = data_at_max_x[['Zn_elem', 'Yn_elem']].values
        values = data_at_max_x[stress_component].values
        grid_z0 = griddata(points, values, (Z_grid, Y_grid), method='cubic')
    
        # Create the contour plot
        contour = ax.contourf(Z_grid, Y_grid, grid_z0 * 1, cmap='magma', levels=100, vmin=global_min * 1, vmax=global_max * 1)
        ax.grid(True, linestyle='--', color='0.8', linewidth=0.5)
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
        
        # Add a textbox with layer name
        ax.text(0.075, 0.5, MyLabels[k], transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.25'))
        
        if k == num_layers - 1:  # Only set xlabel on the last subplot
            ax.set_xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=12)
    
    
    fig.text(0.050, 0.5, 'Depth (mm)', va='center', rotation='vertical', fontweight='bold', fontsize=12)
    
    # Add a color bar
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=global_min * 1, vmax=global_max * 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label(f'{label}'+ ' ' + f'{stress_component}' + ' ' + '(MPa)', fontweight='bold', fontsize=12)
    
    # plt.title(f'Contour Plot of {strain_component} on ZY Plane', fontweight='bold', fontsize=14)
    plt.xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=12)

    # Construct the textbox string with newlines for each piece of information
    textstr = (f"${stress_component}_{{{la}max}}$: {max_stress_value*1:.2f} (MPa)")
    
    # Position the text box in figure coords
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    fig.text(0.95, legend_height, textstr, transform=axs[Structure.index(la)].transAxes, fontsize=10, bbox=props, 
             horizontalalignment='right', verticalalignment='bottom')
    
    plt.grid(True)
    # save figure
    plt.savefig(f'{la}'+ '_' +f'{stress_component}_tire{tstep[0]+idx}_ZYPavSection.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    
    
def plot_SXY_2D(stress_component, dat):

    data = dat.copy()
    data['Xn_elem'] = data['Xn_elem'] - (L/2 - Xw/2)
    
    if stress_component == 'S11':
        label = 'Longitudinal Stress'
        idx = max_S11_idx
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if stress_component == 'S33':
        label = 'Transverse Stress'
        idx = max_S33_idx
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if stress_component == 'S22':
        label = 'Vertical Stress'
        idx = min_S22_idx
        max_stress_value = data[stress_component].min()
        location = data.loc[min_S22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
        
    if stress_component == 'S23':
        label = 'Shear Stress'
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = max_S23_idx
            max_stress_value = data[stress_component].max()
            location = data.loc[max_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = min_S23_idx
            max_stress_value = data[stress_component].min()
            location = data.loc[min_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        legend_height = 0.50
    
    dataframes = process_layers(myData_all[idx].copy())
    num_layers = len(dataframes)
    
    yy_ranges=y_ranges.copy()
    yy_ranges['SG1'] = (yy_ranges['SG1'][1]-500, yy_ranges['SG1'][1])            
                     
    total_y_range = sum(yy_ranges[la][1] - yy_ranges[la][0] for la in Structure)          
    relative_heights = [(yy_ranges[la][1] - yy_ranges[la][0]) / total_y_range for la in Structure]    
    
    fig, axs = plt.subplots(num_layers, 1, figsize=(8, 6), gridspec_kw={'height_ratios': relative_heights}, sharex=True)
    
    plt.subplots_adjust(hspace=0.1)
    
    # Global Parameters
    global_min = min([df[stress_component].min() for df in dataframes])
    global_max = max([df[stress_component].max() for df in dataframes])
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    
    for k, dfs in enumerate(dataframes):
        ax = axs[k] if num_layers>1 else axs
        
        dfs['Xn_elem'] = dfs['Xn_elem'] - (L/2 - Xw/2)
        
        data_at_max_z_raw = dfs[dfs['Zn_elem'] == location['Zn_elem']]
        data_at_max_z = data_at_max_z_raw .sort_values(by=['Yn_elem', 'Node'], ascending=[False, True])
        
        # Create a grid to interpolate onto
        X_unique = np.linspace(data_at_max_z['Xn_elem'].min(), data_at_max_z['Xn_elem'].max(), 1000)
        Y_unique = np.linspace(data_at_max_z['Yn_elem'].min(), data_at_max_z['Yn_elem'].max(), 1000)
        X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
        
        # Interpolate the data onto this grid
        points = data_at_max_z[['Xn_elem', 'Yn_elem']].values
        values = data_at_max_z[stress_component].values
        grid_x0 = griddata(points, values, (X_grid, Y_grid), method='cubic')
        
        # Create the contour plot
        contour = ax.contourf(X_grid, Y_grid, grid_x0 * 1, cmap='magma', levels=100, vmin=global_min * 1, vmax=global_max * 1)
        ax.grid(True, linestyle='--', color='0.8', linewidth=0.5)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
        
        # Add a textbox with layer name
        ax.text(0.075, 0.5, MyLabels[k], transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.25'))

        if k == num_layers - 1:  # Only set xlabel on the last subplot
            ax.set_xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=12)

    fig.text(0.050, 0.5, 'Depth (mm)', va='center', rotation='vertical', fontweight='bold', fontsize=12)    
    
    # Add a color bar
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=global_min * 1, vmax=global_max * 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label(f'{label}'+ ' ' + f'{stress_component}' + ' ' + '(MPa)', fontweight='bold', fontsize=12)
    
    
    #plt.title(f'Contour Plot of {strain_component} on XY Plane', fontweight='bold', fontsize=14)
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=12)
    
    # Construct the textbox string with newlines for each piece of information
    textstr = (f"${stress_component}_{{{la}max}}$: {max_stress_value*1:.2f} MPa")
    
    # Position the text box in figure coords
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    fig.text(0.55, legend_height, textstr, transform=axs[Structure.index(la)].transAxes, fontsize=10, bbox=props, 
             horizontalalignment='right', verticalalignment='bottom')
    
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{stress_component}_tire{tstep[0]+idx}_XYPavSection.png', dpi=500, bbox_inches='tight')
    plt.show()

    
##################################################
########             PLOTS                ######## 
##################################################

plot_E_1DY('E22', myData[min_E22_idx])
plot_E_1DY('E11', myData[max_E11_idx])
plot_E_1DY('E33', myData[max_E33_idx])
plot_E_1DY('E23', myData[min_E23_idx])

plot_E_1DY_multi(['E11', 'E33'], [myData[max_E11_idx], myData[max_E33_idx]])   


plot_EZY_2D('E22', myData[min_E22_idx])
plot_EZY_2D('E11', myData[max_E11_idx])
plot_EZY_2D('E33', myData[max_E33_idx])
plot_EZY_2D('E23', myData[min_E23_idx])

plot_EXY_2D('E22', myData[min_E22_idx])
plot_EXY_2D('E11', myData[max_E11_idx])
plot_EXY_2D('E33', myData[max_E33_idx])
plot_EXY_2D('E23', myData[min_E23_idx])

##############################

plot_S_1DY('S22', myData[min_S22_idx])
plot_S_1DY('S11', myData[max_S11_idx])
plot_S_1DY('S33', myData[max_S33_idx])
plot_S_1DY('S23', myData[min_S23_idx])

plot_SZY_2D('S22', myData[min_S22_idx])
plot_SZY_2D('S11', myData[max_S11_idx])
plot_SZY_2D('S33', myData[max_S33_idx])
plot_SZY_2D('S23', myData[min_S23_idx])

plot_SXY_2D('S22', myData[min_S22_idx])
plot_SXY_2D('S11', myData[max_S11_idx])
plot_SXY_2D('S33', myData[max_S33_idx])
plot_SXY_2D('S23', myData[min_S23_idx])
