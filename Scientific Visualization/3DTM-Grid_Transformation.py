################################################################################
####    Create load input file from Tire Model Output for Pavement Model    ####
####                        Prepared by Angeli Gamez                       #####
################################################################################

from pylab import *
# from scipy import *
# from numpy import *
import numpy as np
# import textwrap
# from matplotlib.path import Path
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import time
import os
# import errno
# import csv
# mpl.style.use('classic')
import pandas as pd
from scipy.interpolate import griddata

start = time.time()

#%%
################################################################################
##################################### INPUT ####################################
################################################################################
SS = ['S1']
PP = ['P1']
VV = ['V1']
RR = ['SL2_3s']#,
      # 'SL6_6s', 'SL6_7s', 'SL6_8s', 'SL6_9s']
## '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s'

# FRAME = -1.

Contact_Length, Contact_Width, Groove = {}, {}, {} # initialize dictionaries
for ss in SS:
    for pp in PP:
        if pp == 'P1':
            P = 26688.0 # Load, [N] = 6.0 [kips] Steer, ICEV
        elif pp == 'P2':
            P = 18904.0 # Load, [N] = 4.2 [kips] Dual, ICEV
        elif pp == 'P3':
            P = 42642.8 # Load, [N] = 9.5 [kips] Steer, EV
        elif pp == 'P4':
            P = 21031.3 # Load, [N] = 4.7 [kips] Dual, EV
        elif pp == 'P5':
            P = 31457.4 # Load, [N] = 7.0 [kips] Steer, EV
        elif pp == 'P6': 
            P = 18823.7 # Load, [N] = 4.2 [kips] Dual, EV
       
        for vv in VV:
            for rr in RR:
                # lc = 'DTA_'+ss+pp+vv+'_'+rr
                lc = pp +'_'+rr#+ '_35to40'
                ## MESH OF THE TIRE MODEL ################################################
                mer = 51 # Tire meridian, number of circumferential lines along tire width
                rib_i = [12, 9, 9, 9, 12] # partition along each rib
                nrib = 5 # Number of ribs
                CS_convert = 1.0 # conversion factor for Michelin load input
                elem_len = 20.0 # element length, [mm]    # Standard:20
                 
                #Path to access input files:
                SPath = 'C:/Users/johan/Box/R27-252 EV/Tasks/Task 3 - Pavement FEM/Input Prep/4. Low-Volume/LV_P1_SL2/stress/P1_SL2/'
                
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
                Nij = np.transpose(df.N.values.reshape(-1,mer))
                Xij = np.transpose(df.X.values.reshape(-1,mer))
                Yij = np.transpose(df.Y.values.reshape(-1,mer))
                Zij = np.transpose(df.Z.values.reshape(-1,mer))
                SSZij = np.transpose(df.CP.values.reshape(-1,mer))
                SSXij = np.transpose(df.CS1.values.reshape(-1,mer))
                SSYij = np.transpose(df.CS2.values.reshape(-1,mer))
                FZij = np.transpose(df.CNF.values.reshape(-1,mer))
                FXij = np.transpose(df.CSF1.values.reshape(-1,mer))
                FYij = np.transpose(df.CSF2.values.reshape(-1,mer))
                Aij = np.transpose(df.CNArea.values.reshape(-1,mer))
                               
                nx_data=len(df['N'])//mer # number of lines in z direction, along contact length

                #Sort according to Yij
                sort_order = np.argsort(Yij[:,0])    # Why?
                Nij = np.array([Nij[i] for i in sort_order])
                Xij = np.array([Xij[i] for i in sort_order])
                Yij = np.array([Yij[i] for i in sort_order])
                SSZij = np.array([SSZij[i] for i in sort_order])
                SSXij = np.array([SSXij[i] for i in sort_order])
                SSYij = np.array([SSYij[i] for i in sort_order])
                FZij = np.array([FZij[i] for i in sort_order])
                FXij = np.array([FXij[i] for i in sort_order])
                FYij = np.array([FYij[i] for i in sort_order])
                Aij = np.array([Aij[i] for i in sort_order])
                        
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
                Xij_trim = -1.0*(Xij_trim-((np.amax(Xij_trim)+np.amin(Xij_trim))/2.0))
                # Yij_norm = -1.0*(Yij_trim-((amax(Yij_trim)+amin(Yij_trim))/2.0))
                              
                # nx_trim = len(ind_nz_nonzero)   
                data_trim_col = np.shape(SSZij_trim)[1] # number of in x direction, along contact length of trimmed data
                
                #%% Calculate Tire Imprint
                #Determine grid geometry (contact width, contact length, and groove between ribs)
                rib_cl = []               # Grid Contact Length
                rib_cw = np.zeros((nrib,data_trim_col))
                y_rib_cw = np.zeros((nrib,data_trim_col))
                rib_cw_avg = []
                groove_w = np.zeros((nrib-1,data_trim_col))
                
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
                y_int = 2.0 #10.25 #Grid coordinate interval [mm]
                
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
                xi = sorted(np.arange(xstart, xstop+x_int, x_int))#reverse=True) 
                ''' 
                    The order does not matter for interpolation ("reverse=True"),
                    but must be checked as it will be needed for generating the 
                    XX and YY text files (last section of code).
                '''
                #xi = arange(xstart, xstop+x_int, x_int)
                
                y_cw_avg = [sum(y_rib_cw[i])/len(y_rib_cw[i]) for i in range(len(y_rib_cw))]
                yi = np.array([y_cw_avg[i] + (cw_max_per_rib[i]/2) - ((j + 1./2.)*cw_max_per_rib[i]/ny) for i in range(nrib) for j in range(ny)])
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
                
                    px = np.zeros((int(b-a),data_trim_col))
                    py = np.zeros((int(b-a),data_trim_col))
                    pssx = np.zeros((int(b-a),data_trim_col))
                    pssy = np.zeros((int(b-a),data_trim_col))
                    pssz = np.zeros((int(b-a),data_trim_col))
                
                    for j in range(b-a):
                        px[j]=Xij_trim[j+a]
                        py[j]=Yij_trim[j+a]
                        
                        pssx[j]=np.array([CS_convert*SSXij_trim[j+a][k] for k in range(len(SSXij_trim[j+a]))])
                        pssy[j]=np.array([CS_convert*SSYij_trim[j+a][k] for k in range(len(SSYij_trim[j+a]))])
                        pssz[j]=np.array([CS_convert*SSZij_trim[j+a][k] for k in range(len(SSZij_trim[j+a]))])
                    
                    for k in range(ny):
                        X.append(xi)
                        Y.append(np.ones(len(xi))*yi[i][k])
                    
                    #X = px.reshape(-1)
                    #Y = py.reshape(-1)
                    # Interpolation using griddata, default method = 'linear'  
                    ssx = (griddata(((px.reshape(-1)),(py.reshape(-1))),(pssx.reshape(-1)),(X,Y), method='linear'))
                    ssy = (griddata(((px.reshape(-1)),(py.reshape(-1))),(pssy.reshape(-1)),(X,Y), method='linear'))
                    ssz = (griddata(((px.reshape(-1)),(py.reshape(-1))),(pssz.reshape(-1)),(X,Y), method='linear'))
                    
                    # Example of grid data methods: ['linear', 'nearest', 'cubic']
                    # Comparing linear to nearest method, linear is more accurate
                        
                    ssx_grid['Rib'+str(i+1)] = np.transpose(ssx)
                    ssy_grid['Rib'+str(i+1)] = np.transpose(ssy)
                    ssz_grid['Rib'+str(i+1)] = np.transpose(ssz)
                
                    ## Check for NaNs
                    Xnan_index['Rib'+str(i+1)] = np.argwhere(np.isnan(ssx_grid['Rib'+str(i+1)]))
                    Ynan_index['Rib'+str(i+1)] = np.argwhere(np.isnan(ssy_grid['Rib'+str(i+1)]))
                    Znan_index['Rib'+str(i+1)] = np.argwhere(np.isnan(ssz_grid['Rib'+str(i+1)]))
                    
                    xnan_ct.append(len(Xnan_index['Rib'+str(i+1)]))
                    ynan_ct.append(len(Ynan_index['Rib'+str(i+1)]))
                    znan_ct.append(len(Znan_index['Rib'+str(i+1)]))
                
                    ## Replace NaNs with zero value
                    '''
                        If the selected mesh is too fine, locate NaNs and replace with zero
                        Usually NaNs arise from being "out-of-the-data-grid"
                        Finely meshing of the interpolation can induce this sensitivity
                    '''
                    where_are_NaNs = np.isnan(ssx)    
                    ssx[where_are_NaNs] = 0
                    ssy[where_are_NaNs] = 0
                    ssz[where_are_NaNs] = 0
                
                    ## Create grid according to each Rib
                    ssx_grid['Rib'+str(i+1)] = np.transpose(ssx)
                    ssy_grid['Rib'+str(i+1)] = np.transpose(ssy)
                    ssz_grid['Rib'+str(i+1)] = np.transpose(ssz)
                
                # Report/ print NaNs per Rib    
                if max(max(xnan_ct),max(ynan_ct),max(znan_ct)) > 0:    
                    print ('COUNT: X NAN =', xnan_ct, '; Y NAN =', ynan_ct, '; Z NAN =', znan_ct)
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
                print('Fine interpolation Difference: fx =', round((((sum(fx_rib)/P))*100.0),2), '%, fy =', round((((sum(fy_rib)/P))*100.0),2), '%, fz =', round((((sum(fz_rib)/P)-1.0)*100.0),2), '%')
                
                #%% Prepare Pavement Model Load Input
                ###########       Create coarse grid for pavement FEM input       #############
                NX = int(cl_global/elem_len)
                NY = 3 # number of partition per rib, along the rib width (for transverse CS) # Standard: 3
                X_INT = elem_len
                
                xstart = -1.0/2.0*cl_global+cl_global/NX/2.0
                xstop = cl_global/2.0-cl_global/NX/2.0
                Xi = sorted(np.arange(xstart, xstop+X_INT, X_INT))#,reverse=True)
                #xi = arange(xstart, xstop+X_INT, X_INT)
                
                y_cw_avg = [sum(y_rib_cw[i])/len(y_rib_cw[i]) for i in range(len(y_rib_cw))]
                Yi_arr = np.array([y_cw_avg[i] - (cw_max_per_rib[i]/2) + ((j + 1./2.)*cw_max_per_rib[i]/NY) for i in range(nrib) for j in range(NY)])
                Yi=Yi_arr.reshape(nrib,NY)
                
                # ## Uncomment the following section to create a uniform grid instead of actual imprint
                # uni_int = 100
                # Yuni_min = myround(min(Yi_arr), uni_int)
                # Yuni_max = myround(max(Yi_arr), uni_int)
                # Yi_arr = array(sorted(arange(Yuni_min,Yuni_max+uni_int,uni_int), reverse=True))
                # Yi=Yi_arr.reshape(1,len(Yi_arr))
                # NY = len(Yi_arr)
                                
                # Create grid of contact stresses
                SX, SY, SZ = {}, {}, {}
                FX_rib, FY_rib, FZ_rib = [], [], []
                
                for i in range(nrib):
                    SSX = np.zeros((NX,NY))
                    SSY = np.zeros((NX,NY))
                    SSZ = np.zeros((NX,NY))
                    for j in range(NX):
                        #print j, 'x range', j*(nx/NX), ((j+1)*(nx/NX))
                        #SSX_I = ssx_grid['Rib'+str(i+1)]
                        #SSX_J = SSX_I[j*(nx/NX):((j+1)*(nx/NX))]        
                        #print shape(SSX_J)
                        for k in range(NY):
                            #print '   y range', k*(ny/NY), ((k+1)*(ny/NY))
                            #SSX_ijk = [SSX_J[l][k*(ny/NY):((k+1)*(ny/NY))] for l in len(SSX_J)]
                            
                            SSX[j,k] = (sum(ssx_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])/np.size((ssx_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])))
                            SSY[j,k] = (sum(ssy_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])/np.size((ssy_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])))
                            SSZ[j,k] = (sum(ssz_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])/np.size((ssz_grid['Rib'+str(i+1)][j*(nx//NX):((j+1)*(nx//NX)),k*(ny//NY):((k+1)*(ny//NY))])))
                
                    SX['Rib'+str(i+1)] = np.transpose(SSX)
                    SY['Rib'+str(i+1)] = np.transpose(SSY)
                    SZ['Rib'+str(i+1)] = np.transpose(SSZ)
                
                    elem_area = (cl_global/NX) * (cw_avg_per_rib[i]/NY)
                    FX_rib.append(sum(SX['Rib'+str(i+1)])*elem_area)
                    FY_rib.append(sum(SY['Rib'+str(i+1)])*elem_area)
                    FZ_rib.append(sum(SZ['Rib'+str(i+1)])*elem_area)
                
                print ('Pavement FEM input Difference: FX =', round((((sum(FX_rib)/P))*100.0),2), '%, FY =', round((((sum(FY_rib)/P))*100.0),2), '%, FZ =', round((((sum(FZ_rib)/P)-1.0)*100.0),2), '%')
                
                # Text files for load input
                txtf = ['SSX', 'SSY', 'SSZ','Mesh']#, 'XX', 'YY']
                
                Contact_Length[lc]=cl_global
                Contact_Width[lc]=cl_global
                Groove[lc]=gw_avg_per_rib
                
                for i in range(len(txtf)):
                    File_exist = os.path.isfile(txtf[i]+'_'+lc+'.txt')
                    if File_exist == True:
                        os.remove(txtf[i]+'_'+lc+'.txt')   
                        OUT = open(txtf[i]+'_'+lc+'.txt', 'w+')
                    else:
                        OUT = open(txtf[i]+'_'+lc+'.txt', 'w+')
                    #OUT.close()
                    #Header
                    
                    if txtf[i] == 'Mesh':
                        OUT.write('Contact Length [mm] =')
                        OUT.write(' ')
                        OUT.write(str(cl_global))
                        OUT.write('\n')
                        OUT.write('Rib Width [mm] = ')
                        for i in range(nrib):
                            OUT.write(str(format(cw_avg_per_rib[i],'.2f')))
                            if i == nrib-1:
                                OUT.write(' ')
                            else:
                                OUT.write(', ')
                        OUT.write('\n')
                        OUT.write('Groove Width [mm] = ')
                        for i in range(nrib-1):
                            OUT.write(str(format(gw_avg_per_rib[i],'.2f')))
                            if i == nrib-1:
                                OUT.write(' ')
                            else:
                                OUT.write(', ')                
                ## DOUBLE CHECK IMPLEMENTATION BEFORE IMPLEMENTING! 09302020 0524AM AJ
                #    elif txtf[i] == 'XX':
                #        XX_txt = Xi #- (min(Xi)-elem_len/2) 
                #        OUT.write('#'+txtf[i])
                #        OUT.write('\n')
                #        for ii in range(NY*nrib):
                #            for jj in range(len(XX_txt)):
                #                OUT.write(str(format(float(XX_txt[jj]),'.5f')))
                #                OUT.write(' ')
                #            OUT.write('\n')
                #
                #    elif txtf[i] == 'YY': 
                #        OUT.write('#'+txtf[i])
                #        OUT.write('\n')
                #        for xx in range(NX):
                #            for ii in range(nrib):
                #                YY_txt = sorted(Yi[ii], reverse=True)
                #                for jj in range(len(YY_txt)):
                #                    OUT.write(str(format(float(YY_txt[jj]),'.5f')))
                #                    OUT.write(' ')
                #            OUT.write('\n')
                                
                    elif txtf[i] == 'SSX':
                        OUT.write('#'+txtf[i])
                        OUT.write('\n')
                        for ii in range(nrib):
                            for jj in range(NY):
                                for kk in range(NX):        
                                    OUT.write(str(format(float(SX['Rib'+str(ii+1)][jj,kk]),'.5f')))
                                    #print str(format(float(SX['Rib'+str(ii+1)][jj,kk]),'.6f'))
                                    OUT.write(' ')
                                OUT.write('\n')
                                
                    elif txtf[i] == 'SSY':
                        OUT.write('#'+txtf[i])
                        OUT.write('\n')
                        for ii in range(nrib):
                            for jj in range(NY):
                                for kk in range(NX):        
                                    OUT.write(str(format(float(SY['Rib'+str(ii+1)][jj,kk]),'.5f')))
                                    #print str(format(float(SY['Rib'+str(ii+1)][jj,kk]),'.6f'))
                                    OUT.write(' ')
                                OUT.write('\n')
                                
                    elif txtf[i] == 'SSZ':
                        OUT.write('#'+txtf[i])
                        OUT.write('\n')
                        for ii in range(nrib):
                            for jj in range(NY):
                                for kk in range(NX):        
                                    OUT.write(str(format(float(SZ['Rib'+str(ii+1)][jj,kk]),'.5f')))
                                    #print str(format(float(SZ['Rib'+str(ii+1)][jj,kk]),'.6f'))
                                    OUT.write(' ')
                                OUT.write('\n')                                                
                
                        #OUT.write('\n')
                    #OUT.write('\n')
                    OUT.close()
################################################################################                    
end = time.time()
elapsed = (end - start)
# print ("Elapsed Time:"), format(elapsed, '.2f'), "seconds."


### References: 
#https://matplotlib.org/examples/pylab_examples/griddata_demo.html