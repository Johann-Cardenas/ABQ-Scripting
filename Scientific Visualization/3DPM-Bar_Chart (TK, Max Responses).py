############################################################################
################       █████   █████████  ███████████       ################ 
################      ░░███   ███░░░░░███░█░░░███░░░█       ################ 
################       ░███  ███     ░░░ ░   ░███  ░        ################ 
################       ░███ ░███             ░███           ################ 
################       ░███ ░███             ░███           ################ 
################       ░███ ░░███     ███    ░███           ################ 
################       █████ ░░█████████     █████          ################ 
################       ░░░░░   ░░░░░░░░░     ░░░░░          ################ 
############################################################################
########        PLOT A BAR CHART USING THE MAXIMUM RESPONSES      ##########
################                MS THESIS                  #################
########        Originally created by: Angeli Jayme              ###########
############################################################################


#__________________________________________________________________________________________________________________________________
from pylab import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import *
import matplotlib.patches as mpatches

#__________________________________________________________________________________________________________________________________

majorLocator = MultipleLocator(20)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(5)

plt.close("all")

## CONFIGURATION OF PLOT SIZE AND MARGINS
plt.rcParams['savefig.dpi']=1000
plt.rcParams['figure.figsize']=3.5,4.5
plt.rcParams['figure.subplot.left']=.17
plt.rcParams['figure.subplot.right']=.965
plt.rcParams['figure.subplot.top']=.935
plt.rcParams['figure.subplot.bottom']=.18

llabel=matplotlib.font_manager.FontProperties(family='Times New Roman', size=7)
bbox_props = dict(boxstyle="square", fc="white", ec="black", lw=2)

width = 0.20       # the width of the bars
normalize = 'N' # 'Y' to produce normalized graphs, otherwise 'N'

# Parse text file (output from post-processing Abaqus ODB)
User='johan'
Path = 'C:/Users/'+User+'/Box/03 MS Thesis/09 Pavement Models/Post-Processing/Steering/'


# SINGLE CONFIGURATION
Case = ['ST_FR_L0_TKS_F', 
        'ST_FA_L1_TKS_S',
        'ST_FA_L1_TKS_W',
        'ST_FA_L2_TKS_S',
        'ST_FA_L2_TKS_W']

l1_e11_plot, l1_e33_plot, l1_e22_plot, l1_e23_min_plot, l1_e23_max_plot, l1_e13_min_plot, l1_e13_max_plot, l1_e12_min_plot, l1_e12_max_plot = {}, {}, {}, {}, {}, {}, {}, {}, {}
l1_s11_plot, l1_s33_plot, l1_s22_plot, l1_s23_min_plot, l1_s23_max_plot, l1_s13_min_plot, l1_s13_max_plot, l1_s12_min_plot, l1_s12_max_plot = {}, {}, {}, {}, {}, {}, {}, {}, {}

for cc in Case:
    E11_data, E33_data, E22_data, E23_min_data, E23_max_data, E13_min_data, E13_max_data, E12_min_data, E12_max_data = {}, {}, {}, {}, {}, {}, {}, {}, {}
    S11_data, S33_data, S22_data, S23_min_data, S23_max_data, S13_min_data, S13_max_data, S12_min_data, S12_max_data = {}, {}, {}, {}, {}, {}, {}, {}, {}
    steps = []
    
    for jj in range(1,18):
        #jj = 15
        Step = 'tire'+str(jj)
        File = Path+'/'+cc+'/Max Responses/'+cc+'_'+Step+'.txt'
        steps.append(Step)
        
        A = open(File, "r")
        Lines = A.readlines() 
        A.close()
        
        XX_ave = eval(Lines[0])
        YY_ave = eval(Lines[1])
        ZZ_ave = eval(Lines[2])
        EE_max = eval(Lines[3])
        EE_min = eval(Lines[4])
        SS_max = eval(Lines[5])
        SS_min = eval(Lines[6])
        
        ## Extracted Strains
        ## MODIFY ACCORDINGLY (THIN OR THICK)
        
        E11 = [EE_max[0][0], EE_max[1][0], EE_max[2][0], EE_max[3][0],EE_max[4][0],EE_max[5][0]]
        E33 = [EE_max[0][2], EE_max[1][2], EE_max[2][2], EE_max[3][2],EE_max[4][2],EE_max[5][2]] 
        E22 = [EE_min[0][1], EE_min[1][1], EE_min[2][1], EE_min[3][1],EE_min[4][1],EE_min[5][1]]
         
        E23_Max = [EE_max[0][5], EE_max[1][5], EE_max[2][5], EE_max[3][5], EE_max[4][5], EE_max[5][5]]
        E23_Min = [EE_min[0][5], EE_min[1][5], EE_min[2][5], EE_min[3][5], EE_min[4][5], EE_min[5][5]]
        
        E13_Max = [EE_max[0][4], EE_max[1][4], EE_max[2][4], EE_max[3][4], EE_max[4][4], EE_max[5][4]]
        E13_Min = [EE_min[0][4], EE_min[1][4], EE_min[2][4], EE_min[3][4], EE_min[4][4], EE_min[5][4]]      
        
        E12_Max = [EE_max[0][3], EE_max[1][3], EE_max[2][3], EE_max[3][3], EE_max[4][3], EE_max[5][3]]
        E12_Min = [EE_min[0][3], EE_min[1][3], EE_min[2][3], EE_min[3][3], EE_min[4][3], EE_min[5][3]]      
        
        
        E11 = np.transpose(E11) * 1E6 #microstrain
        E33 = np.transpose(E33) * 1E6 #microstrain
        E22 = np.transpose(E22) * 1E6 #microstrain
        E23_Min = np.transpose(E23_Min) * 1E6 #microstrain
        E23_Max = np.transpose(E23_Max) * 1E6 #microstrain
    
        E13_Min = np.transpose(E13_Min) * 1E6 #microstrain
        E13_Max = np.transpose(E13_Max) * 1E6 #microstrain
        E12_Min = np.transpose(E12_Min) * 1E6 #microstrain
        E12_Max = np.transpose(E12_Max) * 1E6 #microstrain


        E11_data[Step] = E11
        E33_data[Step] = E33
        E22_data[Step] = E22
        E23_min_data[Step] = E23_Min
        E23_max_data[Step] = E23_Max
        
        E13_min_data[Step] = E13_Min
        E13_max_data[Step] = E13_Max
        E12_min_data[Step] = E12_Min
        E12_max_data[Step] = E12_Max
     
        # ## Extracted Stress
        S11 = [SS_max[0][0], SS_max[1][0], SS_max[2][0], SS_max[3][0], SS_max[4][0], SS_max[5][0]]
        S33 = [SS_max[0][2], SS_max[1][2], SS_max[2][2], SS_max[3][2], SS_max[4][2], SS_max[5][2]] 			   
        S22 = [SS_min[0][1], SS_min[1][1], SS_min[2][1], SS_min[3][1], SS_min[4][1], SS_min[5][1]]
        S23_Min = [SS_max[0][5], SS_max[1][5], SS_max[2][5], SS_max[3][5], SS_max[4][5], SS_max[5][5]]
        S23_Max = [SS_min[0][5], SS_min[1][5], SS_min[2][5], SS_min[3][5], SS_min[4][5], SS_min[5][5]]
           
        S13_Min = [SS_max[0][4], SS_max[1][4], SS_max[2][4], SS_max[3][4], SS_max[4][4], SS_max[5][4]]
        S13_Max = [SS_min[0][4], SS_min[1][4], SS_min[2][4], SS_min[3][4], SS_min[4][4], SS_min[5][4]]
        
        S12_Min = [SS_max[0][3], SS_max[1][3], SS_max[2][3], SS_max[3][3], SS_max[4][3], SS_max[5][3]]
        S12_Max = [SS_min[0][3], SS_min[1][3], SS_min[2][3], SS_min[3][3], SS_min[4][3], SS_min[5][3]]      
        
        
        S11 = np.transpose(S11) #MPa
        S33 = np.transpose(S33) #MPa
        S22 = np.transpose(S22) #MPa
        S23_Min = np.transpose(S23_Min) #MPa
        S23_Max = np.transpose(S23_Max) #MPa
        
        S13_Min = np.transpose(S13_Min) #MPa
        S13_Max = np.transpose(S13_Max) #MPa
        S12_Min = np.transpose(S12_Min) #MPa
        S12_Max = np.transpose(S12_Max) #MPa     
        
        S11_data[Step] = S11
        S33_data[Step] = S33
        S22_data[Step] = S22
        S23_min_data[Step] = S23_Min
        S23_max_data[Step] = S23_Max
     
        S13_min_data[Step] = S13_Min
        S13_max_data[Step] = S13_Max
        S12_min_data[Step] = S12_Min
        S12_max_data[Step] = S12_Max       
        
#__________________________________________________________________________________________________________________________________   
# SELECT LAYER FOR ANALYSIS   

    # Extract from dataframe
    ll = 3
    # 0: Bottom of the Wearing Surface
    # 1: Asphalt Layer [Wearing Surface]
    # 2: Asphalt Layer [Intermediate Surface]
    # 3: Asphalt Layer [Base Layer]
    # 4: Base Course
    # 5: Subgrade
    l1_e11_plot[cc] = [E11_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_e33_plot[cc] = [E33_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_e22_plot[cc] = [abs(E22_data['tire'+str(i)][ll]) for i in range(1,len(steps)+1)]
    l1_e23_min_plot[cc] = [E23_min_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_e23_max_plot[cc] = [E23_max_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    
    l1_e13_min_plot[cc] = [E13_min_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_e13_max_plot[cc] = [E13_max_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]   
    l1_e12_min_plot[cc] = [E12_min_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_e12_max_plot[cc] = [E12_max_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)] 
    
    
    
    l1_s11_plot[cc] = [S11_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_s33_plot[cc] = [S33_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_s22_plot[cc] = [abs(S22_data['tire'+str(i)][ll]) for i in range(1,len(steps)+1)]
    l1_s23_min_plot[cc] = [S23_min_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_s23_max_plot[cc] = [S23_max_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
       
    l1_s13_min_plot[cc] = [S13_min_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_s13_max_plot[cc] = [S13_max_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_s12_min_plot[cc] = [S12_min_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    l1_s12_max_plot[cc] = [S12_max_data['tire'+str(i)][ll] for i in range(1,len(steps)+1)]
    
    
    # Percent difference
    if Case.index(cc) > 0:
        diff_e11_plot = [100.*((l1_e11_plot[Case[0]][i]-l1_e11_plot[cc][i])/l1_e11_plot[Case[0]][i]) for i in range(len(steps))]
        diff_e33_plot = [100.*((l1_e33_plot[Case[0]][i]-l1_e33_plot[cc][i])/l1_e33_plot[Case[0]][i]) for i in range(len(steps))]
        diff_e22_plot = [100.*((l1_e22_plot[Case[0]][i]-l1_e22_plot[cc][i])/l1_e22_plot[Case[0]][i]) for i in range(len(steps))]
        diff_e23_min_plot = [100.*((l1_e23_min_plot[Case[0]][i]-l1_e23_min_plot[cc][i])/l1_e23_min_plot[Case[0]][i]) for i in range(len(steps))]
        diff_e23_max_plot = [100.*((l1_e23_max_plot[Case[0]][i]-l1_e23_max_plot[cc][i])/l1_e23_max_plot[Case[0]][i]) for i in range(len(steps))]
          
        diff_e13_min_plot = [100.*((l1_e13_min_plot[Case[0]][i]-l1_e13_min_plot[cc][i])/l1_e13_min_plot[Case[0]][i]) for i in range(len(steps))]
        diff_e13_max_plot = [100.*((l1_e13_max_plot[Case[0]][i]-l1_e13_max_plot[cc][i])/l1_e13_max_plot[Case[0]][i]) for i in range(len(steps))]       
        diff_e12_min_plot = [100.*((l1_e12_min_plot[Case[0]][i]-l1_e12_min_plot[cc][i])/l1_e12_min_plot[Case[0]][i]) for i in range(len(steps))]
        diff_e12_max_plot = [100.*((l1_e12_max_plot[Case[0]][i]-l1_e12_max_plot[cc][i])/l1_e12_max_plot[Case[0]][i]) for i in range(len(steps))]      
        
            
        diff_s11_plot = [100.*((l1_s11_plot[Case[0]][i]-l1_s11_plot[cc][i])/l1_s11_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s33_plot = [100.*((l1_s33_plot[Case[0]][i]-l1_s33_plot[cc][i])/l1_s33_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s22_plot = [100.*((l1_s22_plot[Case[0]][i]-l1_s22_plot[cc][i])/l1_s22_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s23_min_plot = [100.*((l1_s23_min_plot[Case[0]][i]-l1_s23_min_plot[cc][i])/l1_s23_min_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s23_max_plot = [100.*((l1_s23_max_plot[Case[0]][i]-l1_s23_max_plot[cc][i])/l1_s23_max_plot[Case[0]][i]) for i in range(len(steps))]
                   
        diff_s13_min_plot = [100.*((l1_s13_min_plot[Case[0]][i]-l1_s13_min_plot[cc][i])/l1_s13_min_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s13_max_plot = [100.*((l1_s13_max_plot[Case[0]][i]-l1_s13_max_plot[cc][i])/l1_s13_max_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s12_min_plot = [100.*((l1_s12_min_plot[Case[0]][i]-l1_s12_min_plot[cc][i])/l1_s12_min_plot[Case[0]][i]) for i in range(len(steps))]
        diff_s12_max_plot = [100.*((l1_s12_max_plot[Case[0]][i]-l1_s12_max_plot[cc][i])/l1_s12_max_plot[Case[0]][i]) for i in range(len(steps))]     


#%% LINE PLOTS
# CONFIGURATION OF PLOT SIZE AND MARGINS
matplotlib.rcParams['savefig.dpi']=1000
#    matplotlib.rcParams['figure.figsize']=9.0, 6.5
matplotlib.rcParams['figure.subplot.left']=.055
matplotlib.rcParams['figure.subplot.right']=0.975
matplotlib.rcParams['figure.subplot.top']=0.94
matplotlib.rcParams['figure.subplot.bottom']=0.15
matplotlib.rcParams['font.family'] = 'Times New Roman' # Set font family to Times New Roman
llabel=matplotlib.font_manager.FontProperties(family='Times New Roman', size=20)
plt.ioff() #     
plt.rcParams["figure.autolayout"] = True
plt.rcParams['axes.grid'] = True
mpl.rcParams.update({"axes.grid" : True, "grid.color": "lightgrey", "axes.axisbelow":True})

nrow = 3
ncol = 2
figsize =  (14,10)
fig, ax = plt.subplots(nrow, ncol, figsize=figsize)


label_fs = 10.0
axis_fs = 12.0
# axis_title_sf = 7.5
 
# TANDEM CONFIGURATION
labels = ['Reference (25°C)', 
          'Avg. Weight (45°C)',
          'Avg. Weight (-10°C)',
          'Overweight (45°C)',
          'Overweight (-10°C)']

colors=['#1f77b4', '#5391c9', '#9cb9de', '#e0a7a7', '#cc5f5f']

### LAYER 1 :
x_label = 'Step Number'
legend_loc = 'lower center'

##################################################################################################
##########################                  STRAINS                  #############################
##################################################################################################

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                           ABSOLUTE VALUES
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# ymin_val = 0.
# ymax_val = 50.
# Plot_name = 'Thick_SG_Strain'

# # ----------------------------------------------------------------------------
# ## E11
# # ----------------------------------------------------------------------------

# y_label = 'Longitudinal Tensile Strain'+' ('+r'$\mu \varepsilon_{11}$'+')'
# ax = plt.subplot(nrow,ncol,1)
# ax.plot(l1_e11_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_e11_plot[Case[1]], '^-', label=labels[1], color='red')
# plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')
# # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
# # plt.title( r'$\varepsilon_{11,ac}$')

# # ----------------------------------------------------------------------------
# ## E33
# # ----------------------------------------------------------------------------

# y_label = 'Transverse Tensile Strain'+' ('+r'$\mu \varepsilon_{33}$'+')'
# ax = plt.subplot(nrow,ncol,2)
# ax.plot(l1_e33_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_e33_plot[Case[1]], '^-', label=labels[1], color='red')
# plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## E22 
# # ----------------------------------------------------------------------------

# y_label = 'Vertical Compressive Strain'+' ('+r'$\mu \varepsilon_{22}$'+')'
# ax = plt.subplot(nrow,ncol,3)
# ax.plot(l1_e22_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_e22_plot[Case[1]], '^-', label=labels[1], color='red')
# plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## E23 Max 
# # ----------------------------------------------------------------------------

# y_label = 'Vertical Shear Strain'+' ('+r'$\mu \varepsilon_{23}$'+')'
# ax = plt.subplot(nrow,ncol,4)
# ax.plot(l1_e23_max_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_e23_max_plot[Case[1]], '^-', label=labels[1], color='red')
# plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## E13 Max 
# # ----------------------------------------------------------------------------

# y_label = 'Horizontal Shear Strain'+' ('+r'$\mu \varepsilon_{13}$'+')'
# ax = plt.subplot(nrow,ncol,5)
# ax.plot(l1_e13_max_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_e13_max_plot[Case[1]], '^-', label=labels[1], color='red')
# plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## E12 Max
# # ----------------------------------------------------------------------------

# y_label = 'Transversal Shear Strain'+' ('+r'$\mu \varepsilon_{12}$'+')'
# ax = plt.subplot(nrow,ncol,6)
# ax.plot(l1_e12_max_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_e12_max_plot[Case[1]], '^-', label=labels[1], color='red')
# plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# #_________________________________________________________________________________
# fig.savefig(Plot_name+'.png', dpi=1000)
# close(fig)


# # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# #                           PERCENT DIFFERENCE
# # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
# ymin_val = 0.
# ymax_val = 12.
# Plot_name = 'AC_Strain_Difference'

# # ----------------------------------------------------------------------------
# ## E11
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{11, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,1)
# ax.bar(steps,diff_e11_plot)
# plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Longitudinal Tensile Strain')

# # ----------------------------------------------------------------------------
# ## E33
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{33, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,2)
# ax.bar(steps,diff_e33_plot)
# plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Transverse Tensile Strain')

# # ----------------------------------------------------------------------------
# ## E22
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{22, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,3)
# ax.bar(steps,diff_e22_plot)
# plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Vertical Compressive Strain')

# # ----------------------------------------------------------------------------
# ## E23
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{23, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,4)
# ax.bar(steps,diff_e23_max_plot)
# plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Vertical Shear Strain')

# # ----------------------------------------------------------------------------
# ## E13
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{23, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,5)
# ax.bar(steps,diff_e13_max_plot)
# plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Horizontal Shear Strain')

# # ----------------------------------------------------------------------------
# ## E12
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{23, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,6)
# ax.bar(steps,diff_e12_max_plot)
# plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Transversal Shear Strain')


# #_________________________________________________________________________________
# fig.savefig(Plot_name+'.png', dpi=100)
# close(fig)

##################################################################################################
##########################                   STRESSES                #############################
##################################################################################################


# fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
# ymin_val = 0.
# ymax_val = 3.0
# Plot_name = 'Thick_SG_Stress'


# # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# #                           ABSOLUTE VALUES
# # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# # ----------------------------------------------------------------------------
# ## S11
# # ----------------------------------------------------------------------------

# y_label = 'Longitudinal Tensile Stress (MPa)'
# ax = plt.subplot(nrow,ncol,1)
# ax.plot(l1_s11_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_s11_plot[Case[1]], '^-', label=labels[1], color='red')
# # plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')
# # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
# # plt.title( r'$\varepsilon_{11,ac}$')

# # ----------------------------------------------------------------------------
# ## S33
# # ----------------------------------------------------------------------------

# y_label = 'Transverse Tensile Stress (MPa)'
# ax = plt.subplot(nrow,ncol,2)
# ax.plot(l1_s33_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_s33_plot[Case[1]], '^-', label=labels[1], color='red')
# # plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## S22
# # ----------------------------------------------------------------------------

# y_label = 'Vertical Compressive Stress (MPa)'
# ax = plt.subplot(nrow,ncol,3)
# ax.plot(l1_s22_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_s22_plot[Case[1]], '^-', label=labels[1], color='red')
# # plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## S23
# # ----------------------------------------------------------------------------

# y_label = 'Vertical Shear Stress (MPa)'
# ax = plt.subplot(nrow,ncol,4)
# ax.plot(l1_s23_max_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_s23_max_plot[Case[1]], '^-', label=labels[1], color='red')
# # plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## S13
# # ----------------------------------------------------------------------------

# y_label = 'Horizontal Shear Stress (MPa)'
# ax = plt.subplot(nrow,ncol,5)
# ax.plot(l1_s13_max_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_s13_max_plot[Case[1]], '^-', label=labels[1], color='red')
# # plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')

# # ----------------------------------------------------------------------------
# ## S12
# # ----------------------------------------------------------------------------

# y_label = 'Transversal Shear Stress (MPa)'
# ax = plt.subplot(nrow,ncol,6)
# ax.plot(l1_s12_max_plot[Case[0]], 'o-', label=labels[0], color='green')
# ax.plot(l1_s12_max_plot[Case[1]], '^-', label=labels[1], color='red')
# # plt.ylim([ymin_val, ymax_val])
# plt.legend(fontsize=label_fs, loc=legend_loc,ncol=5, frameon=True, edgecolor='black',fancybox=False)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.grid(color='lightgray', linestyle='--')


# #_____________________________________________________________________________
# fig.savefig(Plot_name+'.png', dpi=1000)
# # plt.close()
# close(fig)

# # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# #                           PERCENT DIFFERENCE
# # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
# ymin_val = 0.
# ymax_val = 15.
# Plot_name = 'AC_Stress_Difference'

# # ----------------------------------------------------------------------------
# ## S11
# # ----------------------------------------------------------------------------

# Plot_name = 'P-401_Stress_Difference'
# # y_label = r'$\varepsilon_{11, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,1)
# ax.bar(steps,diff_s11_plot)
# # plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Longitudinal Tensile Stress')

# # ----------------------------------------------------------------------------
# ## S33
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{33, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,2)
# ax.bar(steps,diff_s33_plot)
# # plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Transverse Tensile Stress')

# # ----------------------------------------------------------------------------
# ## S22
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{22, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,3)
# ax.bar(steps,diff_s22_plot)
# # plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Vertical Compressive Stress')


# # ----------------------------------------------------------------------------
# ## S23
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{23, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,4)
# ax.bar(steps,diff_s23_max_plot)
# # plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Longitudinal Shear Stress')

# # ----------------------------------------------------------------------------
# ## S13
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{23, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,5)
# ax.bar(steps,diff_s13_max_plot)
# # plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Horizontal Shear Stress')

# # ----------------------------------------------------------------------------
# ## S12
# # ----------------------------------------------------------------------------

# # y_label = r'$\varepsilon_{23, P-401}$'+' ('+r'$\mu \varepsilon$'+')'
# y_label = 'Difference (%)'
# ax = plt.subplot(nrow,ncol,6)
# ax.bar(steps,diff_s12_max_plot)
# # plt.ylim([ymin_val, ymax_val])
# # plt.legend(fontsize=label_fs, loc=legend_loc)
# plt.xlabel(x_label, fontsize=axis_fs, fontweight='bold')
# plt.ylabel(y_label, fontsize=axis_fs, fontweight='bold')
# plt.title( 'Transversal Shear Stress')


# #_____________________________________________________________________________
# fig.savefig(Plot_name+'.png', dpi=1000)
# close(fig)



#%% BAR CHARTS
fig_profile, ax_profile = plt.subplots(figsize=(6, 4),facecolor='w')
plt.rcParams['font.family'] = 'Times New Roman'

colors=['green','red','orange','tab:blue','yellow','blue']
hatches=['////','','----','','\\\\\\\\','']

bar_labels=[r'$\varepsilon_{11}$', 
            r'$\varepsilon_{33}$']

x_pos = np.arange(len(bar_labels))
width = 0.9/len(Case)

for t, case in enumerate(Case):
    Response=[max(l1_e11_plot[Case[t]]), 
              max(l1_e33_plot[Case[t]])]
    
    pos = x_pos + t*width - (len(Case)-1)*width/2
    plt.bar(pos, Response, width=width, align='center', alpha=0.6, label=labels[t], color=colors[t], hatch=hatches[t])



plt.xticks(x_pos, bar_labels,fontsize=18, fontdict={'fontname': 'Times New Roman'})
plt.ylabel('Microstrain'+' ('+r'$\mu \varepsilon$'+')', fontsize=18, fontdict={'fontname': 'Times New Roman', 'weight': 'bold'})
plt.yticks(fontsize=18, fontname='Times New Roman')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=14)

plt.grid(color='lightgrey')

plt.savefig("Thick_BottomUP_AC.png",bbox_inches='tight', dpi=1000)
plt.show()


