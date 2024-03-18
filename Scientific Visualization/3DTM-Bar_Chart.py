################################################################################
####    Create load input file from Tire Model Output for Pavement Model    ####
####                        Prepared by Angeli Gamez                       #####
################################################################################

from pylab import *
from scipy import *
from numpy import *
import numpy as np
import textwrap
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import errno
import csv
mpl.style.use('classic')
import pandas as pd
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata


#%%
################################################################################
##################################### INPUT ####################################
################################################################################

fig_profile, ax_profile = plt.subplots(figsize=(3.9, 3.3),dpi=300, facecolor='w')
plt.rcParams['font.family'] = 'Arial'

bar_labels=['Rib1','Rib2','Rib3','Rib4','Rib5']


# Cases=['DTA_3D_S1P2V1_FB',
#        'DTA_3D_S1P4V1_FB',
#        'DTA_3D_S1P6V1_FB',
#        'DTA_3D_S1P8V1_FB',
#        'DTA_3D_S1P10V1_FB',
#        'DTA_3D_S1P3V1_FR']

# labels = ['P2_FB', 
#           'P4_FB', 
#           'P6_FB', 
#           'P8_FB', 
#           'P10_FB',
#           'P3_FR']


Cases=['DTA_3D_S1P2V1_FT',
       'DTA_3D_S1P4V1_FT',
       'DTA_3D_S1P6V1_FT',
       'DTA_3D_S1P8V1_FT',
       'DTA_3D_S1P10V1_FT',
       'DTA_3D_S1P3V1_FR']

labels = ['P2_FT', 
          'P4_FT', 
          'P6_FT', 
          'P8_FT', 
          'P10_FT',
          'P3_FR']


colors = ['yellow','orange','red','tab:blue','green','blue']
hatches=['////','','----','','\\\\\\\\','']


x_pos = np.arange(len(bar_labels))
width = 0.9/len(Cases)

FRAME = -1

Loads=['P2','P4','P6','P8','P10','P3']


for r, case in enumerate(Cases):
    Contact_Length, Contact_Width, Groove = {}, {}, {} # initialize dictionaries
    
    if Loads[r] == 'P1':
        P = 15570.0 # Load, [N]
    elif Loads[r] == 'P2':
        P = 16680.0 # Load, [N]
    elif Loads[r] == 'P3':
        P = 17790.0 # Load, [N]
    elif Loads[r] == 'P4':
        P = 18905.0 # Load, [N]
    elif Loads[r] == 'P5':
        P = 20020.0 # Load, [N]
    elif Loads[r] == 'P6':
        P = 21130.0 # Load, [N]
    elif Loads[r] == 'P7':
        P = 22240.0 # Load, [N]
    elif Loads[r] == 'P8':
        P = 23355.0 # Load, [N]
    elif Loads[r] == 'P9':
        P = 24465.0 # Load, [N]
    elif Loads[r] == 'P10':
        P = 25580.0 # Load, [N]

    lc = Cases[r]
    
    ## MESH OF THE TIRE MODEL ################################################
    mer = 51 # Tire meridian, number of circumferential lines along tire width
    rib_i = [12, 9, 9, 9, 12] # partition along each rib
    nrib = 5 # Number of ribs
    CS_convert = 1.0 # conversion factor for Michelin load input
    elem_len = 10.0 # element length, [mm]    # Standard:20
     
    #Path to access input files:
    SPath = 'C:/Users/johan/Box/03 MS Thesis/08 Tire Models/Processed Profiles/'    
    sub_dir = SPath
    
    #%% Parse Tire FEM Output Text File
    ################################################################################
    def readFile(filename):
        filehandle = open(filename)
        # print filehandle.read()
        filehandle.close()
    
    #Access text file (output of tire model)
    File = os.path.join(sub_dir,lc+'.txt')
    df = pd.read_csv(File, delimiter='\t',index_col=False)   ## DO NOT CONSIDER FIRST COLUMN AS INDEX
    # df.columns = ['N', 'X', 'Y', 'Z', 'CP', 'CS1', 'CS2', 'CNF', 'CSF1', 'CSF2', 'CNArea', 'NT11']
    df.columns = ['N', 'X', 'Y', 'Z', 'CP', 'CS1', 'CS2', 'CNF', 'CSF1', 'CSF2', 'CNArea']
    
    #Create individual dictionaries per variable
    #Data is grouped along each meridian; line of nodes along tire circumference
    Nij = transpose(df.N.values.reshape(-1,mer))
    Xij = transpose(df.X.values.reshape(-1,mer))
    Yij = transpose(df.Y.values.reshape(-1,mer))
    Zij = transpose(df.Z.values.reshape(-1,mer))
    SSZij = transpose(df.CP.values.reshape(-1,mer))
    SSXij = transpose(df.CS1.values.reshape(-1,mer))
    SSYij = transpose(df.CS2.values.reshape(-1,mer))
    FZij = transpose(df.CNF.values.reshape(-1,mer))
    FXij = transpose(df.CSF1.values.reshape(-1,mer))
    FYij = transpose(df.CSF2.values.reshape(-1,mer))
    Aij = transpose(df.CNArea.values.reshape(-1,mer))
                   
    nx_data=len(df['N'])//mer # number of lines in z direction, along contact length

    #Sort according to Yij
    sort_order=argsort(Yij[:,0])    # Why?
    Nij=array([Nij[i] for i in sort_order])
    Xij=array([Xij[i] for i in sort_order])
    Yij=array([Yij[i] for i in sort_order])
    SSZij=array([SSZij[i] for i in sort_order])
    SSXij=array([SSXij[i] for i in sort_order])
    SSYij=array([SSYij[i] for i in sort_order])
    FZij=array([FZij[i] for i in sort_order])
    FXij=array([FXij[i] for i in sort_order])
    FYij=array([FYij[i] for i in sort_order])
    Aij=array([Aij[i] for i in sort_order])
            
    # Find column indices of zero
    idx = np.argwhere(np.all(SSZij[..., :] == 0, axis=0))     
    
    # Delete column of zeros
    Nij_trim = np.delete(Nij, idx, axis=1)
    Xij_trim = np.delete(Xij, idx, axis=1)
    Yij_trim = np.delete(Yij, idx, axis=1)
    Zij_trim = np.delete(Zij, idx, axis=1)
    SSXij_trim = np.delete(SSXij, idx, axis=1)
    SSYij_trim = np.delete(SSYij, idx, axis=1)
    SSZij_trim = np.delete(SSZij, idx, axis=1)
 
    # ## Normalize X and Y coords, centered at zero
    Xij_trim = -1.0*(Xij_trim-((amax(Xij_trim)+amin(Xij_trim))/2.0))
    # Yij_norm = -1.0*(Yij_trim-((amax(Yij_trim)+amin(Yij_trim))/2.0))
                  
    # nx_trim = len(ind_nz_nonzero)   
    data_trim_col = shape(SSZij_trim)[1] # number of in x direction, along contact length of trimmed data
    
    #%% Calculate Tire Imprint
    #Determine grid geometry (contact width, contact length, and groove between ribs)
    rib_cl = []               # Grid Contact Length
    rib_cw = zeros((nrib,data_trim_col))
    y_rib_cw = zeros((nrib,data_trim_col))
    rib_cw_avg = []
    groove_w = zeros((nrib-1,data_trim_col))
    
    for i in range(nrib):
        a = int(sum(rib_i[0:i]))
        b = int(sum(rib_i[0:i+1]))   
        
        for j in range(data_trim_col):
            rib_cw[i][j] = abs(Yij_trim[b-1][j] - Yij_trim[a][j])
            y_rib_cw[i][j] = (Yij_trim[b-1][j] + Yij_trim[a][j])/2.0
            
            if i < nrib-1:
                groove_w[i][j] = abs(Yij_trim[b][j] - Yij_trim[b-1][j])
                                               
        rib_cw_avg.append(sum(rib_cw[i])/len(rib_cw[i]))
    cw_max_per_rib = [max(rib_cw[i]) for i in range(nrib)]
    cw_avg_per_rib = [sum(rib_cw[i])/len(rib_cw[i]) for i in range(nrib)]
    gw_avg_per_rib = [sum(groove_w[i])/len(groove_w[i]) for i in range(nrib-1)]
    
    for i in range(mer):
        rib_cl.append(abs(Xij_trim[i,0]-Xij_trim[i,data_trim_col-1])) #JH Code only references Xij_trim[0] for all 51 sets along contact width
    #%% Interpolate within Tire Output Dataset using Fine Mesh
    #Create fine grid data for later coarse averaging
    #Recall that coarse grid is defined by load input of FEM pavement model (20mm element length, 3 elements per rib along width)
    
    #Grid interval along x- and y-directions
    '''
        Note that these values are iterated manually by user to minimize or best to elimininate NaNs
        NaNs are due to sensitivity of the griddata function to the interpolation coordinate values
        Finer mesh results to higher number of NaNs
       
        When code was verified and checked against JH code, JH assumed all NaNs to be zero wherever found.
        Hence, the motivation for a coarser mesh (still fine within pavement model imprint) to 
        eliminate NaN outputs in the contact stress values.
       
        Main check to consider is force equilibrium in the vertical direction. 09302020
    '''
    x_int = 2.0 #Grid coordinate interval [mm]
    y_int = 2.0#10.25 #Grid coordinate interval [mm]
    
    def myround(x, base):
        return int(base * round(float(x)/base))
    
    Xij_max = myround(max([max(Xij_trim[k]) for k in range(len(Xij))]),x_int)
    Xij_min = myround(min([min(Xij_trim[k]) for k in range(len(Xij))]),x_int)
    
    Yij_max = myround(max([max(Yij_trim[k]) for k in range(len(Xij))]),y_int)
    Yij_min = myround(min([min(Yij_trim[k]) for k in range(len(Xij))]),y_int)
    
    cl_global = float(myround(max(rib_cl),elem_len)) # Global contact length
    nx = int(cl_global / x_int) #number of elements along x, 2-mm element
    ny = int((myround(sum(cw_max_per_rib)+sum(gw_avg_per_rib),elem_len))/y_int)
    
    xstart = -1.0/2.0*cl_global+cl_global/nx/2.0
    xstop = cl_global/2.0-cl_global/nx/2.0
    xi = sorted(arange(xstart, xstop+x_int, x_int))#reverse=True) 
    ''' 
        The order does not matter for interpolation ("reverse=True"),
        but must be checked as it will be needed for generating the 
        XX and YY text files (last section of code).
    '''
    #xi = arange(xstart, xstop+x_int, x_int)
    
    y_cw_avg = [sum(y_rib_cw[i])/len(y_rib_cw[i]) for i in range(len(y_rib_cw))]
    yi = array([y_cw_avg[i] + (cw_max_per_rib[i]/2) - ((j + 1./2.)*cw_max_per_rib[i]/ny) for i in range(nrib) for j in range(ny)])
    yi=yi.reshape(nrib,ny)
    
    ## Using fine interpolation grid, the contact stress values are interpolated
    ## Options in Python: nearest, linear, and cubic as of 09/30/2020
    
    ssx_grid, ssy_grid, ssz_grid = {}, {}, {}
    Xnan_index, Ynan_index, Znan_index = {}, {}, {}
    xnan_ct, ynan_ct, znan_ct = [], [], []
    for i in range(nrib):
        X, Y = [], []
        a = int(sum(rib_i[0:i]))
        b = int(sum(rib_i[0:i+1]))  
    
        px = zeros((int(b-a),data_trim_col))
        py = zeros((int(b-a),data_trim_col))
        pssx = zeros((int(b-a),data_trim_col))
        pssy = zeros((int(b-a),data_trim_col))
        pssz = zeros((int(b-a),data_trim_col))
    
        for j in range(b-a):
            px[j]=Xij_trim[j+a]
            py[j]=Yij_trim[j+a]
            
            pssx[j]=array([CS_convert*SSXij_trim[j+a][k] for k in range(len(SSXij_trim[j+a]))])
            pssy[j]=array([CS_convert*SSYij_trim[j+a][k] for k in range(len(SSYij_trim[j+a]))])
            pssz[j]=array([CS_convert*SSZij_trim[j+a][k] for k in range(len(SSZij_trim[j+a]))])
        
        for k in range(ny):
            X.append(xi)
            Y.append(ones(len(xi))*yi[i][k])
        
        #X = px.reshape(-1)
        #Y = py.reshape(-1)
        # Interpolation using griddata, default method = 'linear'  
        ssx = (griddata(((px.reshape(-1)),(py.reshape(-1))),(pssx.reshape(-1)),(X,Y), method='linear'))
        ssy = (griddata(((px.reshape(-1)),(py.reshape(-1))),(pssy.reshape(-1)),(X,Y), method='linear'))
        ssz = (griddata(((px.reshape(-1)),(py.reshape(-1))),(pssz.reshape(-1)),(X,Y), method='linear'))
        
        # Example of grid data methods: ['linear', 'nearest', 'cubic']
        # Comparing linear to nearest method, linear is more accurate
            
        ssx_grid['Rib'+str(i+1)] = transpose(ssx)
        ssy_grid['Rib'+str(i+1)] = transpose(ssy)
        ssz_grid['Rib'+str(i+1)] = transpose(ssz)
    
        ## Check for NaNs
        Xnan_index['Rib'+str(i+1)] = argwhere(isnan(ssx_grid['Rib'+str(i+1)]))
        Ynan_index['Rib'+str(i+1)] = argwhere(isnan(ssy_grid['Rib'+str(i+1)]))
        Znan_index['Rib'+str(i+1)] = argwhere(isnan(ssz_grid['Rib'+str(i+1)]))
        
        xnan_ct.append(len(Xnan_index['Rib'+str(i+1)]))
        ynan_ct.append(len(Ynan_index['Rib'+str(i+1)]))
        znan_ct.append(len(Znan_index['Rib'+str(i+1)]))
    
        ## Replace NaNs with zero value
        '''
            If the selected mesh is too fine, locate NaNs and replace with zero
            Usually NaNs arise from being "out-of-the-data-grid"
            Finely meshing of the interpolation can induce this sensitivity
        '''
        where_are_NaNs = isnan(ssx)    
        ssx[where_are_NaNs] = 0
        ssy[where_are_NaNs] = 0
        ssz[where_are_NaNs] = 0
    
        ## Create grid according to each Rib
        ssx_grid['Rib'+str(i+1)] = transpose(ssx)
        ssy_grid['Rib'+str(i+1)] = transpose(ssy)
        ssz_grid['Rib'+str(i+1)] = transpose(ssz)
    
    # Report/ print NaNs per Rib    
    if max(max(xnan_ct),max(ynan_ct),max(znan_ct)) > 0:    
        print =('COUNT: X NAN =', xnan_ct, '; Y NAN =', ynan_ct, '; Z NAN =', znan_ct)
    else:
        print ('No NANs!')
    
    #%% Force Equilibirum Check    
    ##Check equilibirum using sum of forces
    '''
        Note that the check is most critical for vertical force, which should
        closely match the applied load (input)
    '''
    fx_rib, fy_rib, fz_rib = [], [], []
    sum_ssx, sum_ssy, sum_ssz = [], [], []
    for i in range(nrib):
        elem_area = (cl_global/nx) * (cw_avg_per_rib[i]/ny)
        fx_rib.append(sum(ssx_grid['Rib'+str(i+1)])*elem_area)
        fy_rib.append(sum(ssy_grid['Rib'+str(i+1)])*elem_area)
        fz_rib.append(sum(ssz_grid['Rib'+str(i+1)])*elem_area)
    
        sum_ssx.append(sum(ssx_grid['Rib'+str(i+1)]))
        sum_ssy.append(sum(ssy_grid['Rib'+str(i+1)]))
        sum_ssz.append(sum(ssz_grid['Rib'+str(i+1)]))             
    
    #%% Prepare Pavement Model Load Input
    ###########       Create coarse grid for pavement FEM input       #############
    NX = int(cl_global/elem_len)
    NY = 4 # number of partition per rib, along the rib width (for transverse CS) # Standard: 3
    X_INT = elem_len
    
    xstart = -1.0/2.0*cl_global+cl_global/NX/2.0
    xstop = cl_global/2.0-cl_global/NX/2.0
    Xi = sorted(arange(xstart, xstop+X_INT, X_INT))#,reverse=True)
    #xi = arange(xstart, xstop+X_INT, X_INT)
    
    y_cw_avg = [sum(y_rib_cw[i])/len(y_rib_cw[i]) for i in range(len(y_rib_cw))]
    Yi_arr = array([y_cw_avg[i] - (cw_max_per_rib[i]/2) + ((j + 1./2.)*cw_max_per_rib[i]/NY) for i in range(nrib) for j in range(NY)])
    Yi=Yi_arr.reshape(nrib,NY)
    
    # Create grid of contact stresses
    SX, SY, SZ = {}, {}, {}
    FX_rib, FY_rib, FZ_rib = [], [], []
    
    for i in range(nrib):
        SSX = zeros((NX,NY))
        SSY = zeros((NX,NY))
        SSZ = zeros((NX,NY))
        for j in range(NX):
            for k in range(NY):   
                SSX[j,k] = (sum(ssx_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])/size((ssx_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])))
                SSY[j,k] = (sum(ssy_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])/size((ssy_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])))
                SSZ[j,k] = (sum(ssz_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])/size((ssz_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])))
    
        SX['Rib'+str(i+1)] = transpose(SSX)
        SY['Rib'+str(i+1)] = transpose(SSY)
        SZ['Rib'+str(i+1)] = transpose(SSZ)
    
        elem_area = (cl_global/NX) * (cw_avg_per_rib[i]/NY)
        FX_rib.append(sum(SX['Rib'+str(i+1)])*elem_area)
        FY_rib.append(sum(SY['Rib'+str(i+1)])*elem_area)
        FZ_rib.append(sum(SZ['Rib'+str(i+1)])*elem_area)
        
    ############################################################################################        
    ############# PLOT #########################################################################  
    pos = x_pos + r*width - (len(Cases)-1)*width/2
    plt.bar(pos, FZ_rib, width=width, align='center', alpha=0.6, label=labels[r], color=colors[r], hatch=hatches[r])



plt.xticks(x_pos, bar_labels,fontsize=12, fontdict={'fontname': 'Arial'})
plt.ylabel('Force (N)', fontsize=12, fontdict={'fontname': 'Arial'})
plt.yticks(fontsize=12, fontname='Arial')


plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
title = ax_profile.set_title('Full Acceleration', pad=10.0, fontsize=13, weight='bold')
plt.grid(color='lightgrey')

#plt.savefig("Bar_Transverse_FB.png",bbox_inches='tight', dpi=1000)
plt.show()

