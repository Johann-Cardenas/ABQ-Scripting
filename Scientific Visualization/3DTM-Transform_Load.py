#############            __     ______     ______            ##############
#############           /\ \   /\  ___\   /\__  _\           ##############
#############           \ \ \  \ \ \____  \/_/\ \/           ##############
#############            \ \_\  \ \_____\    \ \_\           ##############
#############             \/_/   \/_____/     \/_/           ##############                       
   
########              FIT A TIRE IMPRINT TO A GIVEN GRID          #########
################                 MS THESIS                 ################
########            Prepared by: Johann J Cardenas               ##########
###########################################################################

#___________________________________________________________________________
# Load Modules
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
#___________________________________________________________________________
# USER INPUTS

# From Original Simulation
CLength= 220.0  #Contact Length [mm]
ELength=20.0
Rib_Width=[52.84, 41.65, 41.48, 41.61, 52.79]    # Rib Widths [mm]
XElem=int(CLength/ELength)
YElem=[3, 3, 3, 3, 3]

# Target
XTar=12                # Number of Elements Along the Length
ETar=20.0
YTar=[3, 3, 3, 3, 3]    # Partition per Rib
Rib_Tar=[52.5, 41.4, 41.4, 41.4, 52.5]

r=15  # Number of rows
c=12   # Number of columns

S='S1'
P='P7'
V='V1'
Cond='FT'

#___________________________________________________________________________
sV='SSZ'+'_ST_3D_'+S+P+V+'_'+Cond
sL='SSX'+'_ST_3D_'+S+P+V+'_'+Cond
sT='SSY'+'_ST_3D_'+S+P+V+'_'+Cond

CS=[sV, sL, sT]

for t in range(len(CS)):
    #___________________________________________________________________________
    # Check Equilibrium
    
    SSZ_FR= np.loadtxt(CS[t]+'.txt')
    dim=np.array(SSZ_FR.shape)
    
    Rib_Grid=[]
    for i in range(len(YElem)):
        width = Rib_Width[i] / YElem[i]
        Rib_Grid += [width] * YElem[i]
    
    # Force
    Areas = np.zeros((sum(YElem),XElem))
    
    for i in range(sum(YElem)):
        for j in range(XElem):
            Areas[i,j]=ELength*Rib_Grid[i]
    
    Force=SSZ_FR * Areas
    Fi=np.sum(Force);
    print(f"The applied force is: {Fi} kN")
    
    #___________________________________________________________________________
    # Transform into target grid
    
    Rib_Grid_Tar=[]
    for i in range(len(YTar)):
        width = Rib_Tar[i] / YTar[i]
        Rib_Grid_Tar += [width] * YTar[i]
    
    ATar = np.zeros((sum(YTar),XTar))
                    
    for i in range(sum(YTar)):
        for j in range(XTar):
            ATar[i,j]=ETar*Rib_Grid_Tar[i]
    
    # Create the coordinate grid for SSZ_FR
    x_coords = np.linspace(0, CLength, XElem)
    y_coords = np.array([0] + Rib_Grid).cumsum()
    x_grid, y_grid = np.meshgrid(x_coords, y_coords[:-1])
    
    # Create the coordinate grid for SSZ_FR2
    x_new_coords = np.linspace(0, CLength, XTar)
    y_new_coords = np.array([0] + Rib_Grid_Tar).cumsum()
    x_new_grid, y_new_grid = np.meshgrid(x_new_coords, y_new_coords[:-1])
    
    # Interpolate SSZ_FR to the target grid using CubicSpline interpolation
    SSZ_FR2_unscaled = np.zeros((y_new_coords.size - 1, x_new_coords.size))
    for i in range(SSZ_FR.shape[0]):
        cs = CubicSpline(x_coords, SSZ_FR[i, :])
        SSZ_FR2_unscaled[i, :] = cs(x_new_coords)
    
    # Define the objective function to minimize
    def objective_function(scale_factor):
        SSZ_FR2 = SSZ_FR2_unscaled * scale_factor
        return np.abs(np.sum(SSZ_FR2 * ATar) - 1.05*np.sum(SSZ_FR * Areas))
    
    # Minimize the objective function
    initial_guess = 1.0
    result = minimize(objective_function, initial_guess)
    
    # Apply the optimal scaling factor
    optimal_scale_factor = result.x[0]
    SSZ_FR2 = SSZ_FR2_unscaled * optimal_scale_factor
    
    #___________________________________________________________________________
    # Save_File
    
    FTar=np.sum(SSZ_FR2 * ATar)

    if t==1:
        print(f"The applied force in the fitted grid is: {FTar} kN")
        SSZ_FR2_reordered = SSZ_FR2[:, ::-1]
        np.savetxt('sL'+P+'_'+Cond+'.txt', SSZ_FR2_reordered, fmt='%.6f', delimiter='\t', newline='\n')
        
    elif t==2:
        print(f"The applied force in the fitted grid is: {FTar} kN")
        SSZ_FR2_reordered = SSZ_FR2[:, ::-1]
        np.savetxt('sT'+P+'_'+Cond+'.txt', SSZ_FR2_reordered, fmt='%.6f', delimiter='\t', newline='\n')
        
    else:
        print(f"The applied force in the fitted grid is: {FTar} kN")
        np.savetxt('sV'+P+'_'+Cond+'.txt', SSZ_FR2, fmt='%.6f', delimiter='\t', newline='\n') 




