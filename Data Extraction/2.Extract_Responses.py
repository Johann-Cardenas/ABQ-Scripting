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
#      CREATE TEXT FILES OF OUTPUT DATABASE FROM ABAQUS
#       Prepared by: Jaime A. Hernandez & Angeli Jayme
#       Updated by: Johann J. Cardenas (11/28/2023)
#  ----------------------------------------------------------
"""
Previously, we've been extracting nodal outputs using the .odb raw data from the stress/strain/displacement fields.
Because nodal outputs are element-based, this approach would extract only one of the possible nodal outputs (first match).
Under an averaging scheme, we would like to extract all nodal outputs and perform operations on them before writing to file.
This script is a modification of 2.NearSurf.py, where the averaging threshold has been added as a user input.
To run this script, you need to previously extract:
- A .txt file containing all the elements contained in a region of interest, and their connectivities.
- A .txt file containing all the nodes contained in a region of interest, and their coordinates.
The script 1.Sets_ODB can be used to extract these .txt files from the .odb.
"""

# Import necessary packages
from numpy import *
from abaqus import *
from abaqusConstants import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
from  abaqus import session
import sys
import os
import time
from datetime import datetime
import itertools
from collections import OrderedDict

GUI = 0 ## to GUI or not to GUI
def disp(line): # display function
    if GUI:
        print line
    else:
        print >> sys.__stdout__, line

tCumulative = 0
now = datetime.now()

# INPUTs: .odb filename & Path to the .odb file
#-----------------------------------------------
Case = 'CC71DS_P10_AC1W_B1_SB1_SG1'      # Match the name of the .odb file
Group= 'FAA South Section'              
User = 'johannc2'
Path = 'E:/' + Group + '/' + Case + '/'

# INPUTs: Frame number and averaging threshold
#-----------------------------------------------
FRAME = -1
Ave_Threshold = 0.00 # [0.00, 0.25, 0.50, 0.75, 1.00] where 0.00 means no averaging
t_steps = [1 , 16]   # Range of time steps to run the script 

#____________________________________________________________________________________________

disp(Case)
disp('* Start: '+str(now))

# Load element and node dataset outside the loop (These files should be in the same directory as the .odb)
Select_Elem_set, N1, N2, N3, N4, N5, N6, N7, N8 = loadtxt(Case+'_Elem.txt', unpack=True)  
Node_label, x_n, y_n, z_n = loadtxt(Case + '_Nodes.txt', unpack=True)

Select_Node_label = Node_label.astype(int)
Node_label=Node_label.tolist()
Select_Elem_set2 = Select_Elem_set.astype(int)

N1=N1.tolist()
N2=N2.tolist()
N4=N4.tolist()
N5=N5.tolist()

ne=len(Select_Elem_set)
nn=len(Node_label)

Node_Dic = {}
for ii in range(nn):
    Node_Dic[str(int(Node_label[ii]))]=[x_n[ii], y_n[ii], z_n[ii]]


# Calculate element centroid outside inner loops
XXc = [(Node_Dic[str(int(N1[jj]))][0] + Node_Dic[str(int(N2[jj]))][0]) / 2 for jj in range(ne)]
YYc = [(Node_Dic[str(int(N1[jj]))][1] + Node_Dic[str(int(N5[jj]))][1]) / 2 for jj in range(ne)]
ZZc = [(Node_Dic[str(int(N1[jj]))][2] + Node_Dic[str(int(N4[jj]))][2]) / 2 for jj in range(ne)]

# Access Abaqus ODB outside the loop
odb = session.openOdb(name=Path + Case + '.odb', readOnly=True)
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF,))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT,
                                                              refinement=(COMPONENT, 'S11'), )


if Ave_Threshold == 0.00:
    for tt in range(t_steps[0],t_steps[1]): # time step range from 1 to 18
        
        tStart = time.clock()   # Loop time
        STEP='tire'+str(tt)

        Frame = odb.steps[STEP].frames[FRAME]
        Stress=Frame.fieldOutputs['S']
        Strain=Frame.fieldOutputs['E']
        Displacement=Frame.fieldOutputs['U']
        
        if tt == t_steps[0]:
            LocSurf=odb.rootAssembly.instances['PART-1-1'].ElementSetFromElementLabels(name='Block',elementLabels=tuple(Select_Elem_set2))
            LocSurfNodes=odb.rootAssembly.instances['PART-1-1'].NodeSetFromNodeLabels(name='Block',nodeLabels=tuple(Select_Node_label))

        LocSurf=odb.rootAssembly.instances['PART-1-1'].elementSets['Block']
        LocSurfNodes=odb.rootAssembly.instances['PART-1-1'].nodeSets['Block']
        
        Stress_Cen = Stress.getSubset(ELEMENT_NODAL, region=LocSurf)
        Strain_Cen = Strain.getSubset(ELEMENT_NODAL, region=LocSurf)
        Displacement_Cen = Displacement.getSubset(NODAL, region=LocSurfNodes)
        
        # Create dictionaries for quick lookups instead of list.index() method
        EN_nodelist = [Stress_Cen.values[i].nodeLabel for i in range(len(Stress_Cen.values))]
        EN_elemlist = [Stress_Cen.values[i].elementLabel for i in range(len(Stress_Cen.values))]

        NNall = list(itertools.chain(N1, N2, N3, N4, N5, N6, N7, N8)) # collect all nodal connectivities of the elements
        NNres = list(OrderedDict.fromkeys(NNall)) # remove duplicates

        # EN_nodelist_dict = {nodeLabel: i for i, nodeLabel in enumerate(EN_nodelist)}   # Dictionary mapping the first occurance
        EN_nodelist_dict = {nodeLabel: i for i, nodeLabel in reversed(list(enumerate(EN_nodelist)))}

        # Write to File
        with open(Case + '_3DResponse_' + STEP + '.txt', 'w+') as OUT:
            OUT.write('#Elem\tNode\tXn_elem\tYn_elem\tZn_elem\tS11\tS22\tS33\tS23\tE11\tE22\tE33\tE23\tU1\tU2\tU3\tSPmax\tSPmid\tSPmin\tEPmax\tEPmid\tEPmin\tSPress\tSMises\n')
            
            for jj in range(len(Node_label)):
                DNode_label = Displacement_Cen.values[jj].nodeLabel
                Xn_i = x_n[jj]
                Yn_i = y_n[jj]
                Zn_i = z_n[jj]
                
                U1_i = Displacement_Cen.values[jj].data[0]
                U2_i = Displacement_Cen.values[jj].data[1]
                U3_i = Displacement_Cen.values[jj].data[2]
                
                kk = EN_nodelist_dict[DNode_label]
                
                Elem_label = EN_elemlist[kk]

                # Tensor components: ('11', '22', '33', '12', '13', '23')
                
                S11_i = Stress_Cen.values[kk].data[0]
                S22_i = Stress_Cen.values[kk].data[1]
                S33_i = Stress_Cen.values[kk].data[2]
                S23_i = Stress_Cen.values[kk].data[5] 
                
                E11_i = Strain_Cen.values[kk].data[0]
                E22_i = Strain_Cen.values[kk].data[1]
                E33_i = Strain_Cen.values[kk].data[2]
                E23_i = Strain_Cen.values[kk].data[5]

                Smax_i = Stress_Cen.values[kk].maxPrincipal
                Smid_i = Stress_Cen.values[kk].midPrincipal
                Smin_i = Stress_Cen.values[kk].minPrincipal

                Emax_i = Strain_Cen.values[kk].maxPrincipal
                Emid_i = Strain_Cen.values[kk].midPrincipal
                Emin_i = Strain_Cen.values[kk].minPrincipal

                Spres_i = Stress_Cen.values[kk].press
                Smises_i = Stress_Cen.values[kk].mises
                
                OUT.write('{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\n'.format(Elem_label, DNode_label, Xn_i, Yn_i, Zn_i, S11_i, S22_i, S33_i, S23_i, E11_i, E22_i, E33_i, E23_i, U1_i, U2_i, U3_i, Smax_i, Smid_i, Smin_i, Emax_i, Emid_i, Emin_i, Spres_i, Smises_i))
            
            OUT.close()
    
        elapsed1 = (time.clock() - tStart)  # timing
        tCumulative += elapsed1
        disp('* ' + STEP + ' {:.2f} minutes. | Total: {:.2f} minutes.\n'.format(elapsed1 / 60., tCumulative / 60.))

else:
    for tt in range(t_steps[0],t_steps[1]): # time step range from 1 to 18 
        
        tStart = time.clock()   # Loop time
        STEP='tire'+str(tt)

        Frame = odb.steps[STEP].frames[FRAME]
        Stress=Frame.fieldOutputs['S']
        Strain=Frame.fieldOutputs['E']
        Displacement=Frame.fieldOutputs['U']
        
        if tt == t_steps[0]:
            LocSurf=odb.rootAssembly.instances['PART-1-1'].ElementSetFromElementLabels(name='Block',elementLabels=tuple(Select_Elem_set2))
            LocSurfNodes=odb.rootAssembly.instances['PART-1-1'].NodeSetFromNodeLabels(name='Block',nodeLabels=tuple(Select_Node_label))

        LocSurf=odb.rootAssembly.instances['PART-1-1'].elementSets['Block']
        LocSurfNodes=odb.rootAssembly.instances['PART-1-1'].nodeSets['Block']
        
        Stress_Cen = Stress.getSubset(ELEMENT_NODAL, region=LocSurf)
        Strain_Cen = Strain.getSubset(ELEMENT_NODAL, region=LocSurf)
        Displacement_Cen = Displacement.getSubset(NODAL, region=LocSurfNodes)
        
        # Create dictionaries for quick lookups instead of list.index() method
        EN_nodelist = [Stress_Cen.values[i].nodeLabel for i in range(len(Stress_Cen.values))]
        EN_elemlist = [Stress_Cen.values[i].elementLabel for i in range(len(Stress_Cen.values))]

        NNall = list(itertools.chain(N1, N2, N3, N4, N5, N6, N7, N8)) # collect all nodal connectivities of the elements
        NNres = list(OrderedDict.fromkeys(NNall)) # remove duplicates

        EN_nodelist_dict = {}
        for i, nodeLabel in enumerate(EN_nodelist):
            if nodeLabel not in EN_nodelist_dict:
                EN_nodelist_dict[nodeLabel] = [i]  # Start a new list with the index i
            else:
                EN_nodelist_dict[nodeLabel].append(i)  # Append index i to the existing list
        
        EN_nodelist_dict_elem = {nodeLabel: i for i, nodeLabel in reversed(list(enumerate(EN_nodelist)))}   # Only for element labels
        
        # Placeholders for the averaged values
        ave_stress = {}
        ave_strain = {}
        
        ave_smax = {}
        ave_smid = {}
        ave_smin = {}
        ave_emax = {}
        ave_emid = {}
        ave_emin = {}
        ave_spres = {}
        ave_smises = {}
        
        # Write to File
        with open(Case + '_3DResponse_' + STEP + '.txt', 'w+') as OUT:
            OUT.write('#Elem\tNode\tXn_elem\tYn_elem\tZn_elem\tS11\tS22\tS33\tS23\tE11\tE22\tE33\tE23\tU1\tU2\tU3\tSPmax\tSPmid\tSPmin\tEPmax\tEPmid\tEPmin\tSPress\tSMises\n')
            
            for jj in range(len(Node_label)):
                DNode_label = Displacement_Cen.values[jj].nodeLabel
                
                # Get all elements associated with the node label
                element_index = EN_nodelist_dict.get(DNode_label, []) # Gets the index, not the actual element label
                
                # Store all the responses associated with the node label
                stress_components = []
                strain_components = []
                
                smax_components = []
                smid_components = []
                smin_components = []
                emax_components = []
                emid_components = []
                emin_components = []
                spres_components = []
                smises_components = []
                
                kkk = EN_nodelist_dict_elem[DNode_label]  # Only for element labels
            
                stress_components = [Stress_Cen.values[kk].data for kk in element_index]
                strain_components = [Strain_Cen.values[kk].data for kk in element_index]
                
                # Principal stresses and strains
                smax_components = [Stress_Cen.values[kk].maxPrincipal for kk in element_index]
                smid_components = [Stress_Cen.values[kk].midPrincipal for kk in element_index]
                smin_components = [Stress_Cen.values[kk].minPrincipal for kk in element_index]
                emax_components = [Strain_Cen.values[kk].maxPrincipal for kk in element_index]
                emid_components = [Strain_Cen.values[kk].midPrincipal for kk in element_index]
                emin_components = [Strain_Cen.values[kk].minPrincipal for kk in element_index]
                spres_components = [Stress_Cen.values[kk].press for kk in element_index]
                smises_components = [Stress_Cen.values[kk].mises for kk in element_index]
                        
                # AVERAGING
                
                if stress_components:
                    mean_stress = [sum(comp) / len(stress_components) for comp in zip(*stress_components)]
                    valid_stress = [s for s in stress_components if all(abs(s_i - m_i) <= Ave_Threshold * abs(m_i) for s_i, m_i in zip(s, mean_stress))]
                    if valid_stress:
                        ave_stress[DNode_label] = [sum(comp) / len(valid_stress) for comp in zip(*valid_stress)]
                
                if strain_components:
                    mean_strain = [sum(comp) / len(strain_components) for comp in zip(*strain_components)]
                    valid_strain = [e for e in strain_components if all(abs(e_i - m_i) <= Ave_Threshold * abs(m_i) for e_i, m_i in zip(e, mean_strain))]
                    if valid_strain:
                        ave_strain[DNode_label] = [sum(comp) / len(valid_strain) for comp in zip(*valid_strain)]
                
                if smax_components:
                    mean_smax = sum(smax_components) / len(smax_components)
                    valid_smax = [s for s in smax_components if abs(s - mean_smax) <= Ave_Threshold * abs(mean_smax)]
                    if valid_smax:
                        ave_smax[DNode_label] = sum(valid_smax) / len(valid_smax)

                if smid_components:
                    mean_smid = sum(smid_components) / len(smid_components)
                    valid_smid = [s for s in smid_components if abs(s - mean_smid) <= Ave_Threshold * abs(mean_smid)]
                    if valid_smid:
                        ave_smid[DNode_label] = sum(valid_smid) / len(valid_smid)             
        
                if smin_components:
                    mean_smin = sum(smin_components) / len(smin_components)
                    valid_smin = [s for s in smin_components if abs(s - mean_smin) <= Ave_Threshold * abs(mean_smin)]
                    if valid_smin:
                        ave_smin[DNode_label] = sum(valid_smin) / len(valid_smin)   

                if emax_components:
                    mean_emax = sum(emax_components) / len(emax_components)
                    valid_emax = [e for e in emax_components if abs(e - mean_emax) <= Ave_Threshold * abs(mean_emax)]
                    if valid_emax:
                        ave_emax[DNode_label] = sum(valid_emax) / len(valid_emax)   
                        
                if emid_components:
                    mean_emid = sum(emid_components) / len(emid_components)
                    valid_emid = [e for e in emid_components if abs(e - mean_emid) <= Ave_Threshold * abs(mean_emid)]
                    if valid_emid:
                        ave_emid[DNode_label] = sum(valid_emid) / len(valid_emid)

                if emin_components:
                    mean_emin = sum(emin_components) / len(emin_components)
                    valid_emin = [e for e in emin_components if abs(e - mean_emin) <= Ave_Threshold * abs(mean_emin)]
                    if valid_emin:
                        ave_emin[DNode_label] = sum(valid_emin) / len(valid_emin)             
        
                if spres_components:
                    mean_spres = sum(spres_components) / len(spres_components)
                    valid_spres = [s for s in spres_components if abs(s - mean_spres) <= Ave_Threshold * abs(mean_spres)]
                    if valid_spres:
                        ave_spres[DNode_label] = sum(valid_spres) / len(valid_spres)   

                if smises_components:
                    mean_smises = sum(smises_components) / len(smises_components)
                    valid_smises = [s for s in smises_components if abs(s - mean_smises) <= Ave_Threshold * abs(mean_smises)]
                    if valid_smises:
                        ave_smises[DNode_label] = sum(valid_smises) / len(valid_smises)                   
        
                Xn_i = x_n[jj]
                Yn_i = y_n[jj]
                Zn_i = z_n[jj]
                
                # Fetch the averaged values
                U1_i = Displacement_Cen.values[jj].data[0]
                U2_i = Displacement_Cen.values[jj].data[1]
                U3_i = Displacement_Cen.values[jj].data[2]
                
                Elem_label = EN_elemlist[kkk]  # Gets the element label
                
                # Tensor components: ('11', '22', '33', '12', '13', '23')
                
                if DNode_label in ave_stress:
                    S11_i, S22_i, S33_i, S12_i, S13_i, S23_i = ave_stress[DNode_label]
                    
                if DNode_label in ave_strain:
                    E11_i, E22_i, E33_i, E12_i, E13_i, E23_i = ave_strain[DNode_label]

                Smax_i = ave_smax.get(DNode_label, Stress_Cen.values[EN_nodelist_dict_elem[DNode_label]].maxPrincipal)
                Smid_i = ave_smid.get(DNode_label, Stress_Cen.values[EN_nodelist_dict_elem[DNode_label]].midPrincipal)
                Smin_i = ave_smin.get(DNode_label, Stress_Cen.values[EN_nodelist_dict_elem[DNode_label]].minPrincipal)
                Emax_i = ave_emax.get(DNode_label, Strain_Cen.values[EN_nodelist_dict_elem[DNode_label]].maxPrincipal)
                Emid_i = ave_emid.get(DNode_label, Strain_Cen.values[EN_nodelist_dict_elem[DNode_label]].midPrincipal)
                Emin_i = ave_emin.get(DNode_label, Strain_Cen.values[EN_nodelist_dict_elem[DNode_label]].minPrincipal)
                Spres_i = ave_spres.get(DNode_label, Stress_Cen.values[EN_nodelist_dict_elem[DNode_label]].press)
                Smises_i = ave_smises.get(DNode_label, Stress_Cen.values[EN_nodelist_dict_elem[DNode_label]].mises)

                OUT.write('{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\n'.format(Elem_label, DNode_label, Xn_i, Yn_i, Zn_i, S11_i, S22_i, S33_i, S23_i, E11_i, E22_i, E33_i, E23_i, U1_i, U2_i, U3_i, Smax_i, Smid_i, Smin_i, Emax_i, Emid_i, Emin_i, Spres_i, Smises_i))
            OUT.close()
        
        elapsed1 = (time.clock() - tStart)  # timing
        tCumulative += elapsed1
        disp('* ' + STEP + ' {:.2f} minutes. | Total: {:.2f} minutes.\n'.format(elapsed1 / 60., tCumulative / 60.))
