################################################################################
####                     3D SURFACES from CONTACT STRESSES                  ####
####                       Prepared by Johann Cardenas                     #####
################################################################################
    
## VERSION 4
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.tri import Triangulation, TriRefiner
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

Pressure='S1'
Load='P1'
Speed='V1'
STEP = 'FR'
Dir = 'SZ'
case='DTA_'+'3D_'+Pressure+Load+Speed+'_'+STEP

# Load data from text file
Path ='C:/Users/johannc2/Box/03 MS Thesis/08 Tire Models/Outputs/DTA/'+'DTA_'+Pressure+Load+Speed+'/'


df = pd.read_csv(Path+case+'.txt', header=0, usecols=[1, 2, 4], delimiter='\t', names=['X', 'Y', 'Z'])
# usecols=[1, 2, 5]
# 1: 2nd row X Data
# 2: 3nd row Y Data
# 4: 5th row S22 Data (Vertical)
# 5: 6th row S11 Data (Longitudinal)
# 6: 7th row S33 Data (Transverse)

# Find elements with zero values in Z, and remove them from all the columns
zero_positions = np.where(df['Z'] == 0)
df = df.drop(zero_positions[0])


# -----------------------------------------------------------------------
# # CREATE 2D PLOT
# # ----------------------------------------------------------------------
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)

# # Define a regular grid of X and Y points
# xi, yi = np.linspace(df['X'].min(), df['X'].max(), 2500), np.linspace(df['Y'].min(), df['Y'].max(), 2500)
# xi, yi = np.meshgrid(xi, yi)

# # Interpolate the Z values at the regular grid of X and Y points
# zi = griddata((df['X'], df['Y']), df['Z'], (xi, yi),method='linear')

# # Fill in areas with no value with 0
# zi = np.nan_to_num(zi)

# # Create a heatmap with the interpolated Z values
# heatmap = ax2.pcolormesh(xi, yi, zi, cmap=plt.cm.jet)

# # Add a colorbar for the heatmap
# fig2.colorbar(heatmap, shrink=0.5, aspect=5)

# # Set the labels for each axis and adjust their positions
# xlabel = ax2.set_xlabel('Length (mm)', labelpad=7.5)
# ylabel = ax2.set_ylabel('Width (mm)', labelpad=7.5)

# # Set the font weight of the axis labels to bold
# xlabel.set_weight('bold')
# ylabel.set_weight('bold')


# # # Save the figure as a high-resolution PNG file
# # plt.savefig(Path+'2D_'+'DTA_'+Pressure+Load+Speed+'_'+STEP+Dir+'.png', dpi=1000)

# # Show the plot
# plt.show()

########################################################################################

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# Define a regular grid of X and Y points
xi, yi = np.linspace(df['X'].min(), df['X'].max(), 2500), np.linspace(df['Y'].min(), df['Y'].max(), 2500)
xi, yi = np.meshgrid(xi, yi)

# Interpolate the Z values at the regular grid of X and Y points
zi = griddata((df['X'], df['Y']), df['Z'], (xi, yi), method='linear')

# Fill in areas with no value with 0
zi = np.nan_to_num(zi)

# Create a surface plot with the interpolated Z values
surface = ax2.plot_surface(xi, yi, zi, cmap=plt.cm.jet, linewidth=0, antialiased=False)

# Remove ticks and labels from all axes
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_zlabel('')

# Set the aspect ratio to scale down the depth
ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([1, 1, 0.1, 1]))

# Remove the axis lines
ax2.w_xaxis.line.set_visible(False)
ax2.w_yaxis.line.set_visible(False)
ax2.w_zaxis.line.set_visible(False)

# Remove the background color
ax2.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax2.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax2.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

# Remove the gridlines
ax2.grid(False)

# Save the figure as a high-resolution PNG file with transparent background
plt.savefig(Path+'Transparent Contact Stresses', dpi=1000, transparent=True)

# Show the plot
plt.show()


