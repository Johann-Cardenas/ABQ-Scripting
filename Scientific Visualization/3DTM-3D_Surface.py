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
STEP = 'FT'
Dir = 'SY'
case='DTA_'+'3D_'+Pressure+Load+Speed+'_'+STEP

# Load data from text file
Path ='C:/Users/johan/Box/03 MS Thesis/08 Tire Models/Outputs/'+'DTA_'+Pressure+Load+Speed+'/'


df = pd.read_csv(Path+case+'.txt', header=0, usecols=[1, 2, 6], delimiter='\t', names=['X', 'Y', 'Z'])
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
# CREATE 3D PLOT
# ----------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Increase the resolution of the plot by specifying more triangles
triang = Triangulation(df['Y'], df['X'])
cmap = plt.cm.get_cmap('Blues')
facecolors = cmap(np.linspace(0.5, 1, len(triang.triangles)))
facecolors[:, -1] = 0.5  # Set the alpha value of the facecolors to 0.5

# Create the surface plot with filled triangles
surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], triangles=triang.triangles, cmap=plt.cm.jet, linewidth=0.2, facecolors=facecolors)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a colorbar for the surface plot
ax.view_init(30, 70)  # Set the angle of the camera


# Set the labels for each axis and adjust their positions
xlabel = ax.set_xlabel('Width (mm)', labelpad=1)
ylabel = ax.set_ylabel('Length (mm)', labelpad=1)
zlabel = ax.set_zlabel('Stress (MPa)', labelpad=1)

# Set the font weight of the axis labels to bold
xlabel.set_weight('bold')
ylabel.set_weight('bold')
zlabel.set_weight('bold')

# Set the font size of the ticks
ax.tick_params(axis='x', which='major', labelsize=8, pad=1)
ax.tick_params(axis='y', which='major', labelsize=8, pad=1)
ax.tick_params(axis='z', which='major', labelsize=8, pad=1)

# Set the axis limits
ax.set_xlim(-125, 125)
ax.set_ylim(-125, 125)

# Set the grid
ax.grid(True, linestyle='--', color='lightgrey', linewidth=0.25)

# Save the figure as a high-resolution PNG file
plt.savefig(Path+'3D_'+'DTA_'+Pressure+Load+Speed+'_'+STEP+Dir+'.png', dpi=1000)

# Show the plot
plt.show()

# -----------------------------------------------------------------------
# CREATE 2D PLOT
# ----------------------------------------------------------------------
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

# Define a regular grid of X and Y points
xi, yi = np.linspace(df['X'].min(), df['X'].max(), 2500), np.linspace(df['Y'].min(), df['Y'].max(), 2500)
xi, yi = np.meshgrid(xi, yi)

# Interpolate the Z values at the regular grid of X and Y points
zi = griddata((df['X'], df['Y']), df['Z'], (xi, yi),method='linear')

# Fill in areas with no value with 0
zi = np.nan_to_num(zi)

# Create a heatmap with the interpolated Z values
heatmap = ax2.pcolormesh(xi, yi, zi, cmap=plt.cm.jet)

# Add a colorbar for the heatmap
fig2.colorbar(heatmap, shrink=0.5, aspect=5)

# Set the labels for each axis and adjust their positions
xlabel = ax2.set_xlabel('Length (mm)', labelpad=7.5)
ylabel = ax2.set_ylabel('Width (mm)', labelpad=7.5)

# Set the font weight of the axis labels to bold
xlabel.set_weight('bold')
ylabel.set_weight('bold')


# Save the figure as a high-resolution PNG file
plt.savefig(Path+'2D_'+'DTA_'+Pressure+Load+Speed+'_'+STEP+Dir+'.png', dpi=1000)

# Show the plot
plt.show()


#############################################################################################
## VERSION 5 ANIMATION OF 3D PLOT
#############################################################################################

# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from matplotlib.tri import Triangulation
# from matplotlib.animation import FuncAnimation

# # Load data from text file
# data_path ='C:/Users/johannc2/Box/03 MS Thesis/08 Tire Models/Outputs/DTA_S1P1V1/'
# case='DTA_3D_S1P1V1_FR'

# df = pd.read_csv(data_path+case+'.txt', header=0, usecols=[1, 2, 4], delimiter='\t', names=['X', 'Y', 'Z'])

# # Find elements with zero values in Z
# zero_positions = np.where(df['Z'] == 0)

# # Remove elements with zero values in Z from X, Y, and Z
# df = df.drop(zero_positions[0])

# # Create 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Increase the resolution of the plot by specifying more triangles
# triang = Triangulation(df['Y'], df['X'])
# cmap = plt.cm.get_cmap('Blues')
# facecolors = cmap(np.linspace(0.5, 1, len(triang.triangles)))
# facecolors[:, -1] = 0.5  # Set the alpha value of the facecolors to 0.5

# # Create the surface plot with filled triangles
# surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], triangles=triang.triangles, cmap=plt.cm.jet, linewidth=0.2, facecolors=facecolors)

# # Add a colorbar for the surface plot
# fig.colorbar(surf, shrink=0.5, aspect=5)

# # Define a function to update the plot with each frame of the animation
# def update_plot(i):
#     ax.view_init(30, i)  # Set the angle of the camera for the current frame

# # Create the animation
# ani = FuncAnimation(fig, update_plot, frames=range(0, 360, 5), repeat=True)

# # Save the animation as a GIF file
# ani.save('Full Braking.gif', writer='Pillow', fps=10)

# # Show the plot
# plt.show()

