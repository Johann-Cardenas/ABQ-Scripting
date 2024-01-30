"""
1. Extract responses per time step at a specific location
2. Superposition

Created on Wed Jun 22 11:05:59 2022

@author: AJ
"""

# Import libraries
import numpy as np
import pandas as pd
import sys
import os
import time as time # timing
from datetime import datetime
import matplotlib
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from os.path import exists

tStart0 = time.process_time() # Timing, start time
now = datetime.now()

llabel=matplotlib.font_manager.FontProperties(family='Times New Roman', size=8)
bbox_props = dict(boxstyle="square", fc="black", ec="black", lw=2)
matplotlib.rcParams['savefig.dpi']=300
matplotlib.rcParams['figure.figsize']=5.0, 4.0    #6.5, 8.0
matplotlib.rcParams['figure.subplot.left']=.10
matplotlib.rcParams['figure.subplot.right']=.965
matplotlib.rcParams['figure.subplot.top']=.965
matplotlib.rcParams['figure.subplot.bottom']=.12
matplotlib.rcParams['font.family'] = 'Times New Roman'

create_plot = 'Y'

load_cases = ['DTA_FR_L0_TKS_F',
              'DTA_FA_L2_TKS_S',
              'DTA_FB_L2_TKS_S']

naxle = 2 # number of axle(s)
tt_end = [38,38,38]  ## Last time step of ODB


#%% Plotting
    
labels_Plot=['FR',
             'OW_FT',
             'OW_FB']


if create_plot == 'Y':
    # Check if file exists
    User='johan'
    Path2 = 'C:/Users/'+User+'/Box/03 MS Thesis/09 Pavement Models/Post-Processing/DTA/Scripts_Response History/'
    
    depth = [50, 125, 275, 575]#'Depth725mm' ##75, 225, 725
    ## Extract data from summary text files 
    # var_all = ['E11', 'E22', 'E33', 'S22', 'U2']   # Uncomment to get al the responses
    var_all = ['E23']
    
    for dd in depth:
        for var in var_all:
            Plot = []
        
            for lc in load_cases:
                # file_name = 'CC7'+str(naxle)+'DS_'+load_case+'_AC1S_B1_SB1_SG1'+TS+lc+'_'+depth+'_BotLayer2'
                # file_name = 'CC7'+str(naxle)+'DN_P'+str(P)+'_AC8W_B3_SB0_SG3_'+lc+'_Depth'+str(dd)+'mm_BotLayer2'    ### MODIFY
                file_name = lc+'_Depth'+str(dd)+'mm_BotLayer2'    ### MODIFY  
                
                def check_file_exist(Path2, file_name):
                    path_to_file = Path2+file_name+'.txt'
                    file_exists = os.path.exists(path_to_file)
                    return file_exists
                
                
                file_exists = check_file_exist(Path2, file_name)
                if file_exists == True:
                    # np.disp('Yay!')
                    df = pd.read_csv(Path2+file_name+'.txt', sep='\t')
                    if var == var_all[0]:
                        print(file_name+' File exists! Yay!')
                    # print(file_name)
                         
                    # var = 'U2' ## Change to desired response variable
                    Plot.append(np.array(df[var]))
                
            ## formatting changes to spacing
            if var == 'E22':
                matplotlib.rcParams['figure.subplot.left']=.175
                y_label = 'Vertical Strain '+'('+r'$\mu \varepsilon_{22}$'+')'
            if var == 'E11':
                matplotlib.rcParams['figure.subplot.left']=.175
                y_label = 'Longitudinal Strain '+'('+r'$\mu \varepsilon_{11}$'+')'
            if var == 'E33':
                matplotlib.rcParams['figure.subplot.left']=.175
                y_label = 'Transverse Strain '+'('+r'$\mu \varepsilon_{33}$'+')'     
            if var == 'E23':
                matplotlib.rcParams['figure.subplot.left']=.175
                y_label = 'Shear Strain '+'('+r'$\mu \varepsilon_{23}$'+')'        
            if var == 'S22':
                matplotlib.rcParams['figure.subplot.left']=.175
                y_label = 'Vertical Compressive Stress (MPa)'    
            if var == 'U2':
                matplotlib.rcParams['figure.subplot.left']=.175
                y_label = 'Displacement (mm)'
            
            Plot = np.array(Plot).T
            markers = ['-o', '-s', '-^', '-s']
            colors = ['black', 'blue', 'red', 'red']
            
            ## Plot response over time 
            fig = plt.figure()#figsize = (7,7.5))
            x_label = 'Time Step'
            plot_name = 'Thick_'+var+'_Depth'+str(dd)+'mm'
            ax=plt.subplot(1,1,1)

            for ll in range(len(load_cases)):
                plt.plot(np.arange(1,tt_end[ll]), Plot[:,ll]*1000000, markers[ll], color = colors[ll],
                            linewidth=1.0, label =str(labels_Plot[ll]))      
                
            ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey')
            # ax.set_ylim(-500,0)   # Axis Y limit for E22
            ax.yaxis.grid(True, linestyle=':', which='minor', color='gainsboro')
            plt.minorticks_on()          
            ax.xaxis.grid(True, linestyle='--', color='lightgrey')
            plt.ylabel(y_label, fontweight='bold', fontsize=16)
            plt.xlabel(x_label, fontweight='bold', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(loc='best', fontsize=10, ncol=3)      
            # plt.yticks(y_pos, labels=label)
            # plt.xticks(rotation = 0) 
            # plt.xticks(np.arange(1, tt_end[ll], 1))
            # plt.title(plot_name, fontweight='bold')
            fig.savefig(plot_name+'.png', dpi=1000) 
            plt.close('all')