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
#          Surface Plot for Deflection at the Top Layer
#                  By: Johann J Cardenas
# '----------------'  '----------------'  '----------------' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

##################################################
########            User Inputs           ######## 
##################################################

CaseList = ['CC73DS_P5_AC1S_B1_SB1_SG1']
tstep = [34, 35]                          # Range of time steps to be analyzed

# Model Dimensions (Update per case)
L = 35507.0        # Length of the Model
Xw = 4507.0       # Length of the Wheel Path
B = 32750.0       # Width of the Model
Depth = 15000.0    # Total Depth[mm] of the Model

Structure = ['AC1', 'B1', 'SB1', 'SG1']   # Pavement Layers
Thicks = [75.0, 150.0, 500.0, 14275.0]    # Thickness of each layer

user = 'johannc2'
directory = f'C:/Users/{user}/Box/FAA Data Project/04_FEM/00_FEM DATA/FAA_North/{CaseList[0]}/'

# Layer of Analysis
la = 'AC1'

##################################################
########     Preliminary Calculations     ######## 
##################################################

y_ranges = {}
cumulative_thickness = 0
for layer, thickness in zip(Structure, Thicks):
    y_ranges[layer] = (Depth - cumulative_thickness - thickness, Depth - cumulative_thickness)
    cumulative_thickness += thickness

Mylist = []
MyDef = []

for c in CaseList:
    for ts in range(tstep[0], tstep[-1]):
        filename = f'{c}_tire{ts}_3Ddata.txt'
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

        # Append the dataframe to the list
        Mylist.append(df_new)
        MyDef.append(df_new['U2'].min())
      
# Find the index of the minimum value of MyDef
idx = MyDef.index(min(MyDef))
data = Mylist[idx].copy()

##################################################
########       DEFORMATION PLOT           ######## 
##################################################

# Longitudinal Profile Along the Z Axis
def plot_U2(data):

    # Filter the data for the Yn and Zn coordinates that correspond to the surface
    Yn_max = data['Yn_elem'].max()
    filtered_data = data[data['Yn_elem'] == Yn_max].copy()
    
    # Update coordinates to be relative to the wheel path dimensions
    filtered_data['Xn_elem'] = filtered_data['Xn_elem'] - (L/2 - Xw/2)
    filtered_data['Zn_elem'] = filtered_data['Zn_elem'] - (B/2)
    
    # Pivot the data to create a grid for surface plotting
    pivoted_data = filtered_data.pivot_table(index='Zn_elem', columns='Xn_elem', values='U2')

    # Extracting X, Z, and U2 for the plot
    X, Z = np.meshgrid(pivoted_data.columns, pivoted_data.index)
    U2 = pivoted_data.to_numpy()
    
    # Define z-axis limits (Update based on the dataset)
    z_min, z_max = -6.00, -0.70
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Z, U2, cmap='magma', edgecolor='none',vmin=z_min, vmax=z_max )
    #surf = ax.plot_surface(X, Z, U2, cmap='RdYlBu', edgecolor='none',vmin=z_min, vmax=z_max )
    ax.set_zlim(z_min, z_max)

    # Initial view (traffic direction from north to south)
    ax.view_init(elev=30, azim=335)
     
    # Setting tick label font size
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='z', labelsize=9)
    
    # Customizing grid lines
    ax.xaxis._axinfo["grid"].update({"color" : "grey", "linestyle" : ":"})
    ax.yaxis._axinfo["grid"].update({"color" : "grey", "linestyle" : ":"})
    ax.zaxis._axinfo["grid"].update({"color" : "grey", "linestyle" : ":"})

    # Background colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    
    # Add labels and title
    # ax.set_title(f'Surface Deformation (Yn = {Yn_max} mm)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Length (mm)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Width (mm)', fontweight='bold', fontsize=12)
    #ax.set_zlabel(f'Deflection (mm)', fontweight='bold', fontsize=12)

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Deflection (mm)', fontweight='bold', fontsize=12)
    
    # Extract the Xn, Yn coordinates of the maximum deflection
    Xn = filtered_data[filtered_data['U2'] == MyDef[idx]]['Xn_elem'].to_numpy()
    Yn = filtered_data[filtered_data['U2'] == MyDef[idx]]['Zn_elem'].to_numpy()
    
    # Plot a red point at the location of maximum deflection
    ax.scatter(Xn, Yn, MyDef[idx], color='red', s=1, marker='o')

    textstr1 = r'$U_{\max} = ' + f'{MyDef[idx]:.2f}' + r'\ \mathrm{mm}$'
    #textstr2 = f'Time Step = {idx+1:.0f}'
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='black')
    fig.text(0.30, 0.275, s=textstr1, fontsize=9, verticalalignment='bottom', bbox=props)
    #fig.text(0.10, 0.75, s=textstr2, fontsize=9, verticalalignment='bottom', bbox=props)
    
    # Save the figure
    plt.savefig(f'{la}_tire{tstep[0]+idx}_Deflection3D.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    
    
def plot_U2_2D(data):

    # Filter the data for the Yn and Zn coordinates that correspond to the surface
    Yn_max = data['Yn_elem'].max()
    filtered_data = data[data['Yn_elem'] == Yn_max].copy()
    
    # Update coordinates to be relative to the wheel path dimensions
    filtered_data['Xn_elem'] = filtered_data['Xn_elem'] - (L/2 - Xw/2)
    filtered_data['Zn_elem'] = filtered_data['Zn_elem'] - (B/2)
    
    # Pivot the data to create a grid for surface plotting
    pivoted_data = filtered_data.pivot_table(index='Zn_elem', columns='Xn_elem', values='U2')

    # Extracting X, Z, and U2 for the plot
    X, Z = np.meshgrid(pivoted_data.columns, pivoted_data.index)
    U2 = pivoted_data.to_numpy()
    
    # Define z-axis limits (Update based on the dataset)
    z_min, z_max = -6.00, -0.70
    
    # Create the Countour Plot
    fig = plt.figure(figsize=(8, 6))
    
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    contour_filled = plt.contourf(X, Z, U2, cmap='magma', levels=np.linspace(z_min, z_max, 100), norm=norm)
    contour_lines = plt.contour(X, Z, U2, colors='white', linewidths=0.5, levels=np.linspace(z_min, z_max, 20))
    plt.clabel(contour_lines, inline=True, fontsize=12, fmt='%1.1f')
     
    # Add labels and title
    plt.xlabel('Longitudinal Direction (mm)', fontweight='bold', fontsize=14)
    plt.ylabel('Transverse Direction (mm)', fontweight='bold', fontsize=14)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a color bar which maps values to colors
    cbar = plt.colorbar(contour_filled, norm=norm, boundaries=np.linspace(z_min, z_max, 100), ticks=np.linspace(z_min, z_max, 10))
    cbar.set_label('Deflection (mm)', fontweight='bold', fontsize=14)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    textstr1 = r'$U_{\max} = ' + f'{MyDef[idx]:.2f}' + r'\ \mathrm{mm}$'

    # Save the figure
    plt.savefig(f'{la}_tire{tstep[0]+idx}_Deflection2D.png', dpi=500, bbox_inches='tight')
    plt.show()
    

##################################################
########             PLOTS                ######## 
##################################################

plot_U2(data)

plot_U2_2D(data)