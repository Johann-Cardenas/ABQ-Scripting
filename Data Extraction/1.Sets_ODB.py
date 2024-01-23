#############            __     ______     ______            ##############
#############           /\ \   /\  ___\   /\__  _\           ##############
#############           \ \ \  \ \ \____  \/_/\ \/           ##############
#############            \ \_\  \ \_____\    \ \_\           ##############
#############             \/_/   \/_____/     \/_/           ##############                       
   
########   CREATE SETS OF ELEMENTS/NODES FROM OUTPUT DATABASE     #########
################              SINGLE CASES                 ################
########            Prepared by: Johann J Cardenas               ##########
###########################################################################

"""
To run this script '1.Sets_ODB`, you need an .odb file. 
Upon definition of a 'core' region of interest (ROI), the script will output:
- A .txt file containing all the elements contained in the ROI, and their connectivities.
- A .txt file containing all the nodes contained in the ROI, and their coordinates.
"""

from abaqus import *
from abaqusConstants import *
from math import *
from numpy import array
import testUtils
import regionToolset
import time
from odbAccess import *
import numpy as np
from  abaqus import session
from viewerModules import *

testUtils.setBackwardCompatibility()

GUI = 0         # to GUI or not to GUI
def disp(line): # display function
    if GUI:
        print line
    else:
        print >> sys.__stdout__, line

start = time.time()

session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=257, height=178)
session.viewports['Viewport: 1'].maximize()

# INPUTs: .odb filename & Path to the .odb file
#-----------------------------------------------
Case_Name = 'CC71DS_P10_AC1W_B1_SB1_SG1'
Group='FAA South Section'
User = 'johannc2'
ODB_Path = 'E:/' + Group + '/' + Case_Name + '/'

# INPUTs: Model Geometry
#-----------------------------------------------
# Overall Dimensions 
D= 15000.0         # Total Depth of the Model  
L= 33021.0         # Length of the Model
B= 32750.0         # Width of the Model

# Wheelpath Dimensions
X = 2021.0       # Length of the Wheelpath
b = 1750.0         

# Thicknesses
AC = 75.0           # Thickness of the Asphalt Layer  
Base = 150.0        # Thickness of the Base Layer  
Subbase = 500.0     # Thickness of the Subbase Layer  
SG = 500.0          # Depth of Analysis for the Subgrade (Only the first 500mm are considered)

#__________________________________________________________________________________________________________________________________
# Bounding Box
WL = X/2            # Dimension of the Bounding Box in the XZ Plane, from the center of the model.
WT = b/2 + b        # Dimension of the Bounding Box in the YZ Plane, from the center of the model. 
# NOTE: 'WT' Requires extra space to capture maximum shear strain/stresses. Hence, the extra 'b'.

D_tot = AC + Base + Subbase + SG

odb = session.openOdb(name=ODB_Path + Case_Name + '.odb' , readOnly=True)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(UNDEFORMED,))

frame = odb.steps[odb.steps.keys()[-1]].frames[-1]  # Assuming you want the last frame of the last step
instance = odb.rootAssembly.instances['PART-1-1']  # Verify instance name in the odb file

nodes1 = []
elements1 = []

###################################################################################################################################
## BOUNDING BOX
###################################################################################################################################
     
#__________________________________________________________________________________________________________________________________
disp("Processing elements...")
elem_count = len(instance.elements)
for i, elem in enumerate(instance.elements):
    if i % 1000 == 0:
        disp("Progress: {:.2%}".format(i / float(elem_count)))

    elem_nodes = [instance.getNodeFromLabel(node_label) for node_label in elem.connectivity]
    elem_coords = [array(node.coordinates) for node in elem_nodes]
    elem_centroid = np.mean(elem_coords, axis=0)
    
    if L/2 - WL <= elem_centroid[0] <= L/2 + WL and D - D_tot <= elem_centroid[1] <= D + .1 and B/2 - WT <= elem_centroid[2] <= B/2 + WT:    
        elements1.append(elem)

#__________________________________________________________________________________________________________________________________        
disp("Determining the connectivities...")     
unique_connectivities = set()
for elem in elements1:
    connectivity = tuple(sorted([instance.nodes[nodeIdx-1].label for nodeIdx in elem.connectivity]))
    unique_connectivities.add(connectivity)
    
with open('Connectivities.txt', 'w') as f:
    for connectivity_tuple in sorted(unique_connectivities):
        connectivity_list = list(connectivity_tuple)
        f.write(', '.join(map(str, connectivity_list)) + '\n')

unique_node_labels = set()
for connectivity_tuple in unique_connectivities:
    unique_node_labels.update(connectivity_tuple)

#__________________________________________________________________________________________________________________________________
disp("Processing nodes...")
node_count = len(instance.nodes)
for i, node in enumerate(instance.nodes):
    if i % 1000 == 0:
        disp("Progress: {:.2%}".format(i / float(node_count)))

    coords = array(node.coordinates)

    if node.label in unique_node_labels:
        nodes1.append(node)
    
#####################################################################################################################################

#__________________________________________________________________________________________________________________________________
# Export Node Set

disp("Writing Node Set to file...")
with open(Case_Name + '_Nodes.txt', 'w') as OUT:
    OUT.write('#Node\tX\tY\tZ\n')
    for node in nodes1:
        OUT.write('{}\t{}\t{}\t{}\n'.format(node.label, node.coordinates[0], node.coordinates[1], node.coordinates[2]))

#__________________________________________________________________________________________________________________________________
# Export Element Set
disp("Writing Element Set to file...")
OUT2 = open(Case_Name + '_Elem.txt', 'w+')
OUT2.write('#Elem\tN1\tN2\tN3\tN4\tN5\tN6\tN7\tN8\n')
for elem in elements1:
    connectivity = [instance.nodes[nodeIdx-1].label for nodeIdx in elem.connectivity]
    OUT2.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(elem.label, *connectivity))

#__________________________________________________________________________________________________________________________________
disp("Finished. Total time: {:.2f} seconds".format(time.time() - start))
