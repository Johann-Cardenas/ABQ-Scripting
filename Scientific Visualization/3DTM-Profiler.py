################################################################################
####                     2D SURFACES from CONTACT STRESSES                  ####
####                       Prepared by Johann Cardenas                     #####
################################################################################
    
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

Cases=['DTA_3D_S1P1V1_FB',
       'DTA_3D_S1P1V1_FR',
       'DTA_3D_S1P1V1_FT']

fig_profile, ax_profile = plt.subplots(figsize=(4, 3))

# Define a list of colors to use for each dataset
colors = ['red', 'blue', 'green']
labels = ['P1_FB', 'P1_FR', 'P1_FT']

for i, case in enumerate(Cases):

    # Load data from text file
    path = f'C:/Users/johannc2/Box/03 MS Thesis/08 Tire Models/Processed Profiles/{case}.txt'
    
    df = np.loadtxt(path, skiprows=1, usecols=(1, 2, 6), delimiter='\t', unpack=True)
    # 1: 2nd row X Data
    # 2: 3nd row Y Data
    # 4: 5th row S22 Data (Vertical)
    # 5: 6th row S11 Data (Longitudinal)
    # 6: 7th row S33 Data (Transverse)
    
    # Find elements with zero values in Z, and remove them from all the columns
    # df = df[:, df[2] != 0]
    
    # Define a regular grid of X and Y points
    xi, yi = np.linspace(df[0].min(), df[0].max(), 2500), np.linspace(df[1].min(), df[1].max(), 2500)
    xi, yi = np.meshgrid(xi, yi)
    
    # -----------------------------------------------------------------------
    # CREATE STRESS LONGITUDINAL PROFILE
    # ----------------------------------------------------------------------
    
    Ycoord=0
    
    # Interpolate the Z values at the regular grid of X and Y points
    zi = griddata((df[0], df[1]), df[2], (xi, yi), method='cubic')
    
    # Fill in areas with no value with 0
    zi = np.nan_to_num(zi)
    
    # Find the index of the row corresponding to Y=0 in yi
    y0_index = np.argmin(np.abs(yi[:, Ycoord]))
    
    # Extract the Z values for Y=0
    z_profile = zi[y0_index, :]
    
    # Set the font family and size for all text elements
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # Plot the profile with markers and line color set to red
    # ax_profile.plot(xi[y0_index, :], z_profile, marker='o', markersize=4, color='red', label='Profile')
    ax_profile.plot(xi[y0_index, :], z_profile, linewidth=2.0, color=colors[i], label=labels[i])
    
    

# Set the labels for each axis and adjust their positions
xlabel_profile = ax_profile.set_xlabel('Length (mm)', labelpad=7.5)
ylabel_profile = ax_profile.set_ylabel('Stress (MPa)', labelpad=7.5)
    
# Set the font weight of the axis labels to boldd
xlabel_profile.set_weight('bold')
ylabel_profile.set_weight('bold')
    
# Set the lower limit of the Z value axis to 0


ax_profile.set_xlim(-150, 150)
ax_profile.set_xticks([-150, -100, -50, 0, 50, 100, 150])


# ax_profile.set_ylim(0.00, 1.00)   # Vertical
# ax_profile.set_ylim(-0.25, 0.25)   # Longitudinal
ax_profile.set_ylim(-0.10, 0.10)   # Transverse

ax_profile.set_yticks([ 0, 0.25, 0.50, 0.75, 1.00])  # Vertical
# ax_profile.set_yticks([-0.50, -0.25, 0, 0.25, 0.50])  # Longitudinal
# ax_profile.set_yticks([-0.10, -0.05, 0, 0.05, 0.10])  # Transverse

    
# Add minor gridlines with dotted lines
ax_profile.grid(which='major', linestyle='--', linewidth='1.0', color='#D3D3D3')

# Add a legend
legend = ax_profile.legend(loc='upper right',
                           prop={'size': 9, 'family': 'Times New Roman'},
                           ncol=3,
                           columnspacing=1.0,  # Adjust the spacing between columns
                           borderaxespad=0.5,  # Adjust the padding between the legend and the axes
                           handletextpad=0.5)  # Adjust the padding between the handle and the text

# Set the legend frame properties
legend.get_frame().set_facecolor('white')  # Set the background color of the legend box
legend.get_frame().set_edgecolor('black')  # Set the border color of the legend box
legend.get_frame().set_linewidth(0.5)      # Set the border linewidth of the legend box

# Save the file
plt.savefig("Transverse.png", dpi=1000)

# Show the plot
plt.show()

