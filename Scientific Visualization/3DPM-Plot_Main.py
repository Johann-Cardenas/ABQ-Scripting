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
#          1D and 2D Plots for Main Flexible Responses
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
tstep = [10, 16]                          # Range of time steps to be analyzed

# Model Dimensions (Update per case)
L = 32677.0        # Length of the Model
Xw = 1677.0        # Length of the Wheel Path
B = 32750.0        # Width of the Model
Depth = 15000.0    # Total Depth[mm] of the Model|

Structure = ['AC1', 'B1', 'SB1', 'SG1']   # Pavement Layers
Thicks = [75.0, 150.0, 500.0, 14275.0]    # Thickness of each layer

user = 'johan'
directory = f'C:/Users/{user}/Box/FAA Data Project/04_FEM/00_FEM DATA/FAA_South/FAA_South_Responses/{CaseList[0]}/'

# Layer of Analysis
la = 'AC1'

##################################################|
########     Preliminary Calculations     ######## 
##################################################

y_ranges = {}
cumulative_thickness = 0
for layer, thickness in zip(Structure, Thicks):
    y_ranges[layer] = (Depth - cumulative_thickness - thickness, Depth - cumulative_thickness)
    cumulative_thickness += thickness

Mylist = []
listE11, listE22, listE33, listE23max, listE23min, listU2 = [], [], [], [], [], []
listS11, listS22, listS33, listS23max, listS23min = [], [], [], [], []

for c in CaseList:
    for ts in range(tstep[0], tstep[-1]):
        filename = f'{c}_3DResponse_tire{ts}.txt'
        filepath = os.path.join(directory, filename)
        
        # Remove the header
        df = pd.read_csv(filepath, sep='\t')
        
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

        Mylist.append(df_new)
        listU2.append(df_new['U2'].min())
        listE11.append(df_new['E11'].max())
        listE22.append(df_new['E22'].min())
        listE23max.append(df_new['E23'].max())
        listE23min.append(df_new['E23'].min())
        listE33.append(df_new['E33'].max())
        listS11.append(df_new['S11'].max())
        listS22.append(df_new['S22'].min())
        listS23max.append(df_new['S23'].max())
        listS23min.append(df_new['S23'].min())
        listS33.append(df_new['S33'].max())
        
        idxU2 = listU2.index(min(listU2))
        idxE11 = listE11.index(max(listE11))
        idxE22 = listE22.index(min(listE22))
        idxE23max = listE23max.index(max(listE23max))
        idxE23min = listE23min.index(min(listE23min))
        idxE33 = listE33.index(max(listE33))
        idxS11 = listS11.index(max(listS11))
        idxS22 = listS22.index(min(listS22))
        idxS23max = listS23max.index(max(listS23max))
        idxS23min = listS23min.index(min(listS23min))
        idxS33 = listS33.index(max(listS33))

data = Mylist.copy()

##################################################
########        STRAIN EXTRACTION         ######## 
##################################################
 
max_E11_idx = data[idxE11]['E11'].idxmax() # Longitudinal Tensile Strain
min_E22_idx = data[idxE22]['E22'].idxmin() # Vertical Compressive Strain
max_E23_idx = data[idxE23max]['E23'].idxmax() # Maximum Shear Strain
min_E23_idx = data[idxE23min]['E23'].idxmin() # Minimum Shear Strain
max_E33_idx = data[idxE33]['E33'].idxmax() # Transverse Tensile Strain
min_U2_idx = data[idxU2]['U2'].idxmin() # Vertical Displacement

# Retrieve the location (Xn, Yn, Zn) for each maximum value

max_E11_location = data[idxE11].loc[max_E11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
min_E22_location = data[idxE22].loc[min_E22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
max_E23_location = data[idxE23max].loc[max_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
min_E23_location = data[idxE23min].loc[min_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
max_E33_location = data[idxE33].loc[max_E33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
min_U2_location = data[idxU2].loc[min_U2_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]

##################################################
########        STRESS EXTRACTION         ######## 
##################################################

max_S11_idx = data[idxS11]['S11'].idxmax() # Longitudinal Tensile Stress
min_S22_idx = data[idxS22]['S22'].idxmin() # Vertical Compressive Stress
max_S23_idx = data[idxS23max]['S23'].idxmax() # Maximum Shear Stress
min_S23_idx = data[idxS23min]['S23'].idxmin() # Minimum Shear Stress
max_S33_idx = data[idxS33]['S33'].idxmax() # Transverse Tensile Stress

# Retrieve the location (Xn, Yn, Zn) for each maximum value
max_S11_location = data[idxS11].loc[max_S11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
min_S22_location = data[idxS22].loc[min_S22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
max_S23_location = data[idxS23max].loc[max_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
min_S23_location = data[idxS23min].loc[min_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
max_S33_location = data[idxS33].loc[max_S33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]

#%%

##################################################
########           STRAIN PLOTS           ######## 
##################################################

#%%
# ____________________________________________________________________________________________________
# Longitudinal Profile Along the Z Axis

def plot_E_Z(strain_component, dat):
    
    data = dat.copy()
    data['Zn_elem'] = data['Zn_elem'] - (B/2)
    
    if strain_component == 'E11':
        label = 'Longitudinal Strain'
        idx = idxE11
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.30
        
    if strain_component == 'E33':
        label = 'Transverse Strain'
        idx = idxE33
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.30  
        
    if strain_component == 'E22':
        label = 'Vertical Strain'
        idx = idxE22
        max_strain_value = data[strain_component].min()
        location = data.loc[min_E22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.30
        
    if strain_component == 'E23':
        label = 'Shear Strain'
    
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = idxE23max
            max_strain_value = data[strain_component].max()
            location = data.loc[max_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxE23min
            max_strain_value = data[strain_component].min()
            location = data.loc[min_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.15
        
    # Filter the data for the Yn and Zn coordinates that correspond to the maximum strain value
    filtered_data = data[(data['Yn_elem'] == location['Yn_elem']) & (data['Xn_elem'] == location['Xn_elem'])]
    # Sort the filtered data based on the Xn coordinate
    filtered_data_sorted = filtered_data.sort_values('Zn_elem')    
        
    # Create the line plot
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(filtered_data_sorted['Zn_elem'], filtered_data_sorted[strain_component]*1000000, color='b', marker='o', markersize =4)
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    plt.plot(location['Zn_elem'], max_strain_value*1000000, color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    
    #Plot an arrow to the maximum strain value
    plt.annotate('', xy=(location['Zn_elem'], max_strain_value*1000000), xytext=(location['Zn_elem'], max_strain_value*1000000 + 35),
                 arrowprops=dict(facecolor='red', shrink=0.04))
    
    #plt.title(f'Longitudinal Profile of {strain_component} Along Zn Axis', fontweight='bold', fontsize=14)
    plt.xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel(f'{label}' + ' ' + f'{strain_component}' + ' ' + '$(\mu\epsilon)$', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    textstr = (f"${strain_component}_{{max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$\n"
               f"$X$: {location['Xn_elem'] - (L/2-Xw/2)}" + f' ' +
               f"$Y$: {location['Zn_elem']}" + f' ' +
               f"$Z$: {Depth - location['Yn_elem']}")
    # Position the text box in figure coords to ensure it's always in the same position regardless of data
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.gcf().text(0.60, textbox_height, textstr, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_1DZ.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    
# ____________________________________________________________________________________________________
# Longitudinal Profile Along the X Axis

def plot_E_X(strain_component, dat):

    data = dat.copy()
    data['Xn_elem'] = data['Xn_elem'] - (L/2 - Xw/2)

    if strain_component == 'E11':
        label = 'Longitudinal Strain'
        idx = idxE11
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.40
        textbox_loc = 0.40
        
    if strain_component == 'E33':
        label = 'Transverse Strain'
        idx = idxE33
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.40
        textbox_loc = 0.40  
        
    if strain_component == 'E22':
        label = 'Vertical Strain'
        idx = idxE22
        max_strain_value = data[strain_component].min()
        location = data.loc[min_E22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.20  
        textbox_loc = 0.40
        
    if strain_component == 'E23':
        label = 'Shear Strain'
        
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = idxE23max
            max_strain_value = data[strain_component].max()
            location = data.loc[max_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxE23min
            max_strain_value = data[strain_component].min()
            location = data.loc[min_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.20  
        textbox_loc = 0.40
        
    # Filter the data for the Yn and Zn coordinates that correspond to the maximum strain value
    filtered_data = data[(data['Yn_elem'] == location['Yn_elem']) & (data['Zn_elem'] == location['Zn_elem'])]
    # Sort the filtered data based on the Xn coordinate
    filtered_data_sorted = filtered_data.sort_values('Xn_elem')    
        
    # Create the line plot
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(filtered_data_sorted['Xn_elem'], filtered_data_sorted[strain_component]*1000000, color='b', marker='o')
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    plt.plot(location['Xn_elem'], max_strain_value*1000000, color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    
    #Plot an arrow to the maximum strain value
    plt.annotate('', xy=(location['Xn_elem'], max_strain_value*1000000), xytext=(location['Xn_elem'], max_strain_value*1000000 + 35),
                 arrowprops=dict(facecolor='red', shrink=0.04))
    
    #plt.title(f'Longitudinal Profile of {strain_component} Along Xn Axis', fontweight='bold', fontsize=14)
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel(f'{label}' + ' ' + f'{strain_component}' + ' ' + '$(\mu\epsilon)$', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    textstr = (f"${strain_component}_{{max}}$: {max_strain_value*1000000:.2f} $(\mu\epsilon)$\n"
               f"$X$: {location['Xn_elem']}" + f' ' +
               f"$Y$: {location['Zn_elem'] - B/2 }" + f' ' +
               f"$Z$: {Depth - location['Yn_elem']}")
    
    # Position the text box in figure coords to ensure it's always in the same position regardless of data
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.gcf().text(textbox_loc, textbox_height, textstr, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_1DX.png', dpi=500, bbox_inches='tight')
    plt.show()


# ____________________________________________________________________________________________________
# Longitudinal Cut Along the ZY Plane

def plot_E_ZY(strain_component, dat):
    
    data = dat.copy()
    data['Zn_elem'] = data['Zn_elem'] - (B/2)
    
    if strain_component == 'E11':
        label = 'Longitudinal Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -1000.0, 1000.0 , 1000000.0, 10
        idx = idxE11
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    elif strain_component == 'E33':
        label = 'Transverse Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -1000.0, 1000.0 , 1000000.0, 10
        idx = idxE33
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    elif strain_component == 'E22':
        label = 'Vertical Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -400.0, 500.0, 1000000.0, 7
        idx = idxE22
        max_strain_value = data[strain_component].min()
        location = data.loc[min_E22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    elif strain_component == 'E23':
        label = 'Shear Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -500.0, 500.0, 1000000.0, 5
        
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = idxE23max
            max_strain_value = data[strain_component].max()
            location = data.loc[max_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxE23min
            max_strain_value = data[strain_component].min()
            location = data.loc[min_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
            
    elif strain_component == 'U2':
        label = 'Deflection' +' '+ f'{strain_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -8.0, 2.0, 1.0, 10 
        idx = idxU2
        max_strain_value = data[strain_component].min()
        location = data.loc[min_U2_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    data_at_max_x = data[data['Xn_elem'] == location['Xn_elem']]
    
    # Create a grid to interpolate onto
    Z_unique = np.linspace(data_at_max_x['Zn_elem'].min(), data_at_max_x['Zn_elem'].max(), 500)
    Y_unique = np.linspace(data_at_max_x['Yn_elem'].min(), data_at_max_x['Yn_elem'].max(), 500)
    Z_grid, Y_grid = np.meshgrid(Z_unique, Y_unique)
    
    # Interpolate the data onto this grid
    points = data_at_max_x[['Zn_elem', 'Yn_elem']].values
    values = data_at_max_x[strain_component].values
    grid_z0 = griddata(points, values, (Z_grid, Y_grid), method='cubic')
    
    # Create the contour plot
    plt.figure(figsize=(6, 4), dpi=300)

    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    contour_filled = plt.contourf(Z_grid, Y_grid, grid_z0 * k, cmap='magma', levels=np.linspace(z_min, z_max, 100), norm=norm)
    contour_lines = plt.contour(Z_grid, Y_grid, grid_z0 * k, colors='white', linewidths=0.5, levels=np.linspace(z_min, z_max, clines))
    plt.clabel(contour_lines, inline=True, fontsize=11, fmt='%d')
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
    
    plt.plot(location['Zn_elem'], location['Yn_elem'], color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Calculate offsets as a percentage of the axis range
    x_offset = (xlims[1] - xlims[0]) * 0.1
    y_offset = (ylims[1] - ylims[0]) * 0.1
    
    # Calculate the new position for the textbox
    textbox_x = location['Zn_elem'] + x_offset
    textbox_y = location['Yn_elem'] + y_offset
    
    # Add a color bar
    cbar = plt.colorbar(contour_filled, norm=norm, boundaries=np.linspace(z_min, z_max, 100), ticks=np.linspace(z_min, z_max, 5))
    cbar.set_label(f'{label}', fontweight='bold', fontsize=14)
    
    plt.xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Construct the textbox string with newlines for each piece of information
    if strain_component == 'U2':
        textstr = (f"${strain_component}_{{max}}$: {max_strain_value:.2f} mm")
    else:
        textstr = (f"${{{strain_component}}}_{{max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$")
    
    # Position the text box in figure coords
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.text(textbox_x, textbox_y, textstr, fontsize=12, bbox=props, 
             horizontalalignment='right', verticalalignment='bottom', zorder=5)
    
    # Add an arrow point to the location of the maximum strain
    plt.annotate('', xy=(location['Zn_elem'], location['Yn_elem']), xytext=(location['Zn_elem'], location['Yn_elem'] + (y_ranges[la][1]-y_ranges[la][0])/15),
                 arrowprops=dict(facecolor='red', shrink=0.04))
    
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{strain_component}_tire{tstep[0]+idx}_2D-YZ.png', dpi=500, bbox_inches='tight')
    plt.show()
   
# ____________________________________________________________________________________________________    
# Longitudinal Cut Along the XY Plane

def plot_E_XY(strain_component, dat):
   
    data = dat.copy()
    data['Xn_elem'] = data['Xn_elem'] - (L/2 - Xw/2)
    
    if strain_component == 'E11':
        label = 'Longitudinal Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -500.0, 500.0 , 1000000.0, 10
        idx = idxE11
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    elif strain_component == 'E33':
        label = 'Transverse Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -500.0, 500.0 , 1000000.0, 15
        idx = idxE33
        max_strain_value = data[strain_component].max()
        location = data.loc[max_E33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    elif strain_component == 'E22':
        label = 'Vertical Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -500.0, 500.0, 1000000.0, 10
        idx = idxE22
        max_strain_value = data[strain_component].min()
        location = data.loc[min_E22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    elif strain_component == 'E23':
        label = 'Shear Strain' +' '+ f'{strain_component}'+' '+'$(\mu\epsilon)$'
        z_min, z_max, k, clines = -500.0, 250.0, 1000000.0, 10
        
        if abs(data[strain_component].max()) >= abs(data[strain_component].min()):
            idx = idxE23max
            max_strain_value = data[strain_component].max()
            location = data.loc[max_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxE23min
            max_strain_value = data[strain_component].min()
            location = data.loc[min_E23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
            
    elif strain_component == 'U2':
        label = 'Deflection' +' '+ f'{strain_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -8.0, 0.0, 1.0, 15 
        idx = idxU2
        max_strain_value = data[strain_component].min()
        location = data.loc[min_U2_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
            
    
    data_at_max_z = data[data['Zn_elem'] == location['Zn_elem']]
    
    # Create a grid to interpolate onto
    X_unique = np.linspace(data_at_max_z['Xn_elem'].min(), data_at_max_z['Xn_elem'].max(), 500)
    Y_unique = np.linspace(data_at_max_z['Yn_elem'].min(), data_at_max_z['Yn_elem'].max(), 500)
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
    
    # Interpolate the data onto this grid
    points = data_at_max_z[['Xn_elem', 'Yn_elem']].values
    values = data_at_max_z[strain_component].values
    grid_x0 = griddata(points, values, (X_grid, Y_grid), method='cubic')
    
    # Create the contour plot
    plt.figure(figsize=(6, 4), dpi=300)

    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    contour_filled  = plt.contourf(X_grid, Y_grid, grid_x0 * k, cmap='magma', levels=np.linspace(z_min, z_max, 100), norm=norm)
    contour_lines = plt.contour(X_grid, Y_grid, grid_x0 * k, colors='white', linewidths=0.5, levels=np.linspace(z_min, z_max, clines))
    plt.clabel(contour_lines, inline=True, fontsize=11, fmt='%d')
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
    
    plt.plot(location['Xn_elem'], location['Yn_elem'], color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    # Calculate offsets as a percentage of the axis range
    x_offset = (xlims[1] - xlims[0]) * 0.1
    y_offset = (ylims[1] - ylims[0]) * 0.1
    
    # Calculate the new position for the textbox
    textbox_x = location['Xn_elem'] + x_offset
    textbox_y = location['Yn_elem'] + y_offset
    
    # Add a color bar
    cbar = plt.colorbar(contour_filled, norm=norm, boundaries=np.linspace(z_min, z_max, 100), ticks=np.linspace(z_min, z_max, 5) )
    cbar.set_label(f'{label}', fontweight='bold', fontsize=14)
    
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Construct the textbox string with newlines for each piece of information
    if strain_component == 'U2':
        textstr = (f"${strain_component}_{{max}}$: {max_strain_value:.2f} mm")
    else:
        textstr = (f"${{{strain_component}}}_{{max}}$: {max_strain_value*1000000:.2f} $\mu\epsilon$")
    
    # Position the text box in figure coords
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.text(textbox_x, textbox_y, textstr, fontsize=12, bbox=props, 
             horizontalalignment='right', verticalalignment='bottom', zorder=5)
    
    
    plt.annotate('', xy=(location['Xn_elem'], location['Yn_elem']), xytext=(location['Xn_elem'], location['Yn_elem'] + (y_ranges[la][1]-y_ranges[la][0])/15),
                 arrowprops=dict(facecolor='red', shrink=0.04))
    
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' + f'{strain_component}_tire{tstep[0]+idx}_2D-XY.png', dpi=500, bbox_inches='tight')
    plt.show()
    

##################################################
########           STRESS PLOTS           ######## 
##################################################   
    
# ____________________________________________________________________________________________________
# Longitudinal Profile Along the Z Axis

def plot_S_Z(stress_component, dat):

    data = dat.copy()
    data['Zn_elem'] = data['Zn_elem'] - (B/2)

    if stress_component == 'S11':
        label = 'Longitudinal Stress'
        idx = idxS11
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.30
        
    if stress_component == 'S33':
        label = 'Transverse Stress'
        idx = idxS33
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.30  
        
    if stress_component == 'S22':
        label = 'Vertical Stress'
        idx = idxS22
        max_stress_value = data[stress_component].min()
        location = data.loc[min_S22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.30
        
    if stress_component == 'S23':
        label = 'Shear Stress'
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = idxS23max
            max_stress_value = data[stress_component].max()
            location = data.loc[max_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxS23min
            max_stress_value = data[stress_component].min()
            location = data.loc[min_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.15
        
    # Filter the data for the Yn and Zn coordinates that correspond to the maximum strain value
    filtered_data = data[(data['Yn_elem'] == location['Yn_elem']) & (data['Xn_elem'] == location['Xn_elem'])]
    # Sort the filtered data based on the Xn coordinate
    filtered_data_sorted = filtered_data.sort_values('Zn_elem')    
        
    # Create the line plot
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(filtered_data_sorted['Zn_elem'], filtered_data_sorted[stress_component]*1, color='b', marker='o', markersize =4)
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    plt.plot(location['Zn_elem'], max_stress_value, color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    
    #Plot an arrow to the maximum strain value
    plt.annotate('', xy=(location['Zn_elem'], max_stress_value), xytext=(location['Zn_elem'], max_stress_value + (data[stress_component].max()-data[stress_component].min())/15),
                 arrowprops=dict(facecolor='red', shrink=0.04))
    
    #plt.title(f'Longitudinal Profile of {stress_component} Along Zn Axis', fontweight='bold', fontsize=14)
    plt.xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel(f'{label}' + ' ' + f'{stress_component}' + ' ' + '(MPa)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    textstr = (f"${stress_component}_{{max}}$: {max_stress_value*1:.2f} MPa\n"
               f"$X$: {location['Xn_elem'] - (L/2-Xw/2)}" + f' ' +
               f"$Y$: {location['Zn_elem']}" + f' ' +
               f"$Z$: {Depth - location['Yn_elem']:.1f}")
    
    # Position the text box in figure coords to ensure it's always in the same position regardless of data
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.gcf().text(0.60, textbox_height, textstr, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{stress_component}_tire{tstep[0]+idx}_1DZ.png', dpi=500, bbox_inches='tight')
    plt.show()
    
# ____________________________________________________________________________________________________
# Longitudinal Profile Along the X Axis

def plot_S_X(stress_component, dat):

    data = dat.copy()
    data['Xn_elem'] = data['Xn_elem'] - (L/2 - Xw/2)

    if stress_component == 'S11':
        label = 'Longitudinal Stress'
        idx = idxS11
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.40
        textbox_loc = 0.40
        
    if stress_component == 'S33':
        label = 'Transverse Stress'
        idx = idxS33
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.40
        textbox_loc = 0.40  
        
    if stress_component == 'S22':
        label = 'Vertical Stress'
        idx = idxS22
        max_stress_value = data[stress_component].min()
        location = data.loc[min_S22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.20  
        textbox_loc = 0.40
        
    if stress_component == 'S23':
        label = 'Shear Stress'
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = idxS23max
            max_stress_value = data[stress_component].max()
            location = data.loc[max_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxS23min
            max_stress_value = data[stress_component].min()
            location = data.loc[min_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        textbox_height = 0.20  
        textbox_loc = 0.40
        
    # Filter the data for the Yn and Zn coordinates that correspond to the maximum strain value
    filtered_data = data[(data['Yn_elem'] == location['Yn_elem']) & (data['Zn_elem'] == location['Zn_elem'])]
    # Sort the filtered data based on the Xn coordinate
    filtered_data_sorted = filtered_data.sort_values('Xn_elem')    
        
    # Create the line plot
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(filtered_data_sorted['Xn_elem'], filtered_data_sorted[stress_component]*1, color='b', marker='o')
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    plt.plot(location['Xn_elem'], max_stress_value, color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    
    #Plot an arrow to the maximum strain value
    plt.annotate('', xy=(location['Xn_elem'], max_stress_value), xytext=(location['Xn_elem'], max_stress_value + (data[stress_component].max()-data[stress_component].min())/15),
                 arrowprops=dict(facecolor='red', shrink=0.04))

    #plt.title(f'Longitudinal Profile of {stress_component} Along Xn Axis', fontweight='bold', fontsize=14)
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel(f'{label}' + ' ' + f'{stress_component}' + ' ' + '(MPa)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    textstr = (f"${stress_component}_{{max}}$: {max_stress_value*1:.2f} MPa\n"
               f"$X$: {location['Xn_elem']}" + f' ' +
               f"$Y$: {location['Zn_elem'] - B/2 }" + f' ' +
               f"$Z$: {Depth - location['Yn_elem']:.1f}")
    
    # Position the text box in figure coords to ensure it's always in the same position regardless of data
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.gcf().text(textbox_loc, textbox_height, textstr, fontsize=12, bbox=props, horizontalalignment='center', verticalalignment='bottom')
    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{stress_component}_tire{tstep[0]+idx}_1DX.png', dpi=500, bbox_inches='tight')
    plt.show()

# ____________________________________________________________________________________________________
# Longitudinal Cut Along the ZY Plane

def plot_S_ZY(stress_component, dat):

    data = dat.copy()
    data['Zn_elem'] = data['Zn_elem'] - (B/2)
    
    if stress_component == 'S11': 
        label = 'Longitudinal Stress' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -10.0, 10.0, 1.0, 5
        idx = idxS11
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S33':
        label = 'Transverse Stress' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0, 1.0, 5
        idx = idxS33
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S22':
        label = 'Vertical Stress' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0, 1.0, 5
        idx = idxS22
        max_stress_value = data[stress_component].min()
        location = data.loc[min_S22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S23':
        label = 'Shear Stress' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0, 1.0, 5
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = idxS23max
            max_strain_value = data[stress_component].max()
            location = data.loc[max_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxS23min
            max_stress_value = data[stress_component].min()
            location = data.loc[min_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]

    data_at_max_x = data[data['Xn_elem'] == location['Xn_elem']]
    
    # Create a grid to interpolate onto
    Z_unique = np.linspace(data_at_max_x['Zn_elem'].min(), data_at_max_x['Zn_elem'].max(), 500)
    Y_unique = np.linspace(data_at_max_x['Yn_elem'].min(), data_at_max_x['Yn_elem'].max(), 500)
    Z_grid, Y_grid = np.meshgrid(Z_unique, Y_unique)
    
    # Interpolate the data onto this grid
    points = data_at_max_x[['Zn_elem', 'Yn_elem']].values
    values = data_at_max_x[stress_component].values
    grid_z0 = griddata(points, values, (Z_grid, Y_grid), method='cubic')
    
    # Create the contour plot
    plt.figure(figsize=(6, 4), dpi=300)

    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    contour_filled = plt.contourf(Z_grid, Y_grid, grid_z0 * k, cmap='magma', levels=np.linspace(z_min, z_max, 100), norm=norm)
    contour_lines = plt.contour(Z_grid, Y_grid, grid_z0 * k, colors='white', linewidths=0.5, levels=np.linspace(z_min, z_max, clines))
    plt.clabel(contour_lines, inline=True, fontsize=11, fmt='%d')
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)

    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
    
    plt.plot(location['Zn_elem'], location['Yn_elem'], color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    # Calculate offsets as a percentage of the axis range
    x_offset = (xlims[1] - xlims[0]) * 0.1
    y_offset = (ylims[1] - ylims[0]) * 0.1
    
    # Calculate the new position for the textbox
    textbox_x = location['Zn_elem'] + x_offset
    textbox_y = location['Yn_elem'] + y_offset
    
    # Add a color bar
    cbar = plt.colorbar(contour_filled, norm=norm, boundaries=np.linspace(z_min, z_max, 100), ticks=np.linspace(z_min, z_max, 5))
    cbar.set_label(f'{label}', fontweight='bold', fontsize=14)
    
    plt.xlabel('Transverse Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Construct the textbox string with newlines for each piece of information
    textstr = (f"${{{stress_component}}}_{{max}}$: {max_stress_value*1:.2f} (MPa)")
    
    # Position the text box in figure coords
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.text(textbox_x, textbox_y, textstr, transform=plt.gca().transAxes, fontsize=12, bbox=props, 
             horizontalalignment='center', verticalalignment='bottom', zorder=5)
    
    # Add an arrow point to the location of the maximum strain
    plt.annotate('', xy=(location['Zn_elem'], location['Yn_elem']), xytext=(location['Zn_elem'], location['Yn_elem'] + (y_ranges[la][1]-y_ranges[la][0])/15),
                 arrowprops=dict(facecolor='red', shrink=0.04))

    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' +f'{stress_component}_tire{tstep[0]+idx}_2D-YZ.png', dpi=500, bbox_inches='tight')
    plt.show()
    
# ____________________________________________________________________________________________________   
# Longitudinal Cut Along the XY Plane

def plot_S_XY(stress_component, dat):
    
    data = dat.copy()
    data['Xn_elem'] = data['Xn_elem'] - (L/2 - Xw/2)
    
    if stress_component == 'S11':
        label = 'Longitudinal Strain' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0 , 1.0, 10
        idx = idxS11
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S11_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S33':
        label = 'Transverse Stress' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0 , 1.0, 10
        idx = idxS33
        max_stress_value = data[stress_component].max()
        location = data.loc[max_S33_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S22':
        label = 'Vertical Strain' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0 , 1.0, 10
        idx = idxS22
        max_stress_value = data[stress_component].min()
        location = data.loc[min_S22_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        
    if stress_component == 'S23':
        label = 'Shear Stress' +' '+ f'{stress_component}'+' '+'$(mm)$'
        z_min, z_max, k, clines = -5.0, 5.0 , 1.0, 10
        
        if abs(data[stress_component].max()) >= abs(data[stress_component].min()):
            idx = idxS23max
            max_stress_value = data[stress_component].max()
            location = data.loc[max_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
        else:
            idx = idxS23min
            max_stress_value = data[stress_component].min()
            location = data.loc[min_S23_idx, ['Xn_elem', 'Yn_elem', 'Zn_elem']]
    
    data_at_max_z = data[data['Zn_elem'] == location['Zn_elem']]
    
    # Create a grid to interpolate onto
    X_unique = np.linspace(data_at_max_z['Xn_elem'].min(), data_at_max_z['Xn_elem'].max(), 500)
    Y_unique = np.linspace(data_at_max_z['Yn_elem'].min(), data_at_max_z['Yn_elem'].max(), 500)
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
    
    # Interpolate the data onto this grid
    points = data_at_max_z[['Xn_elem', 'Yn_elem']].values
    values = data_at_max_z[stress_component].values
    grid_x0 = griddata(points, values, (X_grid, Y_grid), method='cubic')
    
    # Create the contour plot
    plt.figure(figsize=(6, 4), dpi=300)
    
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    contour_filled  = plt.contourf(X_grid, Y_grid, grid_x0 * k, cmap='magma', levels=np.linspace(z_min, z_max, 100), norm=norm)
    contour_lines = plt.contour(X_grid, Y_grid, grid_x0 * k, colors='white', linewidths=0.5, levels=np.linspace(z_min, z_max, clines))
    plt.clabel(contour_lines, inline=True, fontsize=11, fmt='%d')
    plt.grid(True, linestyle='--', color='0.8', linewidth=0.5)
    
    def subtract(x):
        return f"{Depth - x:.0f}"
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: subtract(x)))
    
    plt.plot(location['Xn_elem'], location['Yn_elem'], color='green', marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    # Calculate offsets as a percentage of the axis range
    x_offset = (xlims[1] - xlims[0]) * 0.1
    y_offset = (ylims[1] - ylims[0]) * 0.1
    
    # Calculate the new position for the textbox
    textbox_x = location['Xn_elem'] + x_offset
    textbox_y = location['Yn_elem'] + y_offset
    
    # Add a color bar
    cbar = plt.colorbar(contour_filled, norm=norm, boundaries=np.linspace(z_min, z_max, 100), ticks=np.linspace(z_min, z_max, 5) )
    cbar.set_label(f'{label}', fontweight='bold', fontsize=14)
    
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel('Depth (mm)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Construct the textbox string with newlines for each piece of information
    textstr = (f"${{{stress_component}}}_{{max}}$: {max_stress_value*1:.2f} (MPa)")
    
    # Position the text box in figure coords
    props = dict(boxstyle="round4,pad=0.5", edgecolor='black', facecolor='white', linewidth=2)
    plt.text(textbox_x, textbox_y, textstr, transform=plt.gca().transAxes, fontsize=12, bbox=props, 
             horizontalalignment='right', verticalalignment='bottom')
    
    plt.annotate('', xy=(location['Xn_elem'], location['Yn_elem']), xytext=(location['Xn_elem'], location['Yn_elem'] + (y_ranges[la][1]-y_ranges[la][0])/15),
                 arrowprops=dict(facecolor='red', shrink=0.04))

    plt.grid(True)
    
    plt.savefig(f'{la}'+ '_' + f'{stress_component}_tire{tstep[0]+idx}_2D-XY.png', dpi=500, bbox_inches='tight')
    plt.show()
    
##################################################
########             PLOTS                ######## 
##################################################

plot_E_Z('E22', data[idxE22])
plot_E_Z('E11', data[idxE11])
plot_E_Z('E23', data[idxE23min])
plot_E_Z('E33', data[idxE33])

#___________________________________________________________________________

plot_E_ZY('E22', data[idxE22])
plot_E_ZY('E11', data[idxE11])
plot_E_ZY('E23', data[idxE23min])
plot_E_ZY('E33', data[idxE33])
plot_E_ZY('U2', data[idxU2])

#___________________________________________________________________________

plot_E_X('E22', data[idxE22])
plot_E_X('E11', data[idxE11])
plot_E_X('E23', data[idxE23min])
plot_E_X('E33', data[idxE33])

#___________________________________________________________________________

plot_E_XY('E22', data[idxE22])
plot_E_XY('E11', data[idxE11])
plot_E_XY('E23', data[idxE23min])
plot_E_XY('E33', data[idxE33])
plot_E_XY('U2', data[idxU2])

## NOTE: Notice that for E23, one has to choose between idxE23max and E23min.
# Lines 140-176 will output these values in the console to verify the choosing. 

##############################

plot_S_Z('S22', data[idxS22])
plot_S_Z('S11', data[idxS11])
plot_S_Z('S23', data[idxS23min])
plot_S_Z('S33', data[idxS33])

#___________________________________________________________________________

plot_S_ZY('S22', data[idxS22])
plot_S_ZY('S11', data[idxS11])
plot_S_ZY('S23', data[idxS23min])
plot_S_ZY('S33', data[idxS33])

#___________________________________________________________________________

plot_S_X('S22', data[idxS22])
plot_S_X('S11', data[idxS11])
plot_S_X('S23', data[idxS23min])
plot_S_X('S33', data[idxS33])

#___________________________________________________________________________

plot_S_XY('S22', data[idxS22])
plot_S_XY('S11', data[idxS11])
plot_S_XY('S23', data[idxS23min])
plot_S_XY('S33', data[idxS33])

#___________________________________________________________________________