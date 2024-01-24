################################################################################
####    Extract Information from the Foot Region of the Tire Model          ####
####                        Prepared by Angeli Gamez                       #####
################################################################################

from abaqus import *
from abaqusConstants import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
from abaqus import session
import sys
import time
t0= time.clock()

Pressure='S1'
Load='P7'
Speed='V1'
Case='ST_3D_'+Pressure+Load+Speed

Tempcase = 'HE'
Path='C:/Users/johan/Box/03 MS Thesis/08 Tire Models/Outputs/Steering/'+'ST_'+Pressure+Load+Speed+'/'

STEP = 'FB' ## FR: Free Rolling;    FB: Full Braking  ;   FT: Full Acceleration

FRAME_COUNTER = -1  # Value by default

Ele_Sets=['PART-1-1.FOOT']
Nod_Sets=['FOOT']

print >> sys.__stdout__, Case
odb = session.openOdb(name=Path+Case+'.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_UNDEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='CPRESS', outputPosition=ELEMENT_NODAL, )

OUT1 = open(Case+'_'+STEP+'_CNORMFsum.txt','w+')
OUT1.write('#Inc\t')
OUT1.write('#StepTime\t')
OUT1.write('CNORMF_Sum\n')
    
# for FRAME in odb.steps[STEP].frames:
# FRAME= odb.steps[STEP].frames[FRAME[kk]]

FRAME= odb.steps[STEP].frames[FRAME_COUNTER]
Disp=FRAME.fieldOutputs['U']
C_Press=FRAME.fieldOutputs['CPRESS']
C_Shear1=FRAME.fieldOutputs['CSHEAR1']
C_Shear2=FRAME.fieldOutputs['CSHEAR2']
C_NormF=FRAME.fieldOutputs['CNORMF']
C_ShearF=FRAME.fieldOutputs['CSHEARF']
C_NArea=FRAME.fieldOutputs['CNAREA']
# Cener=FRAME.fieldOutputs['CENER']     #Only for Visco Hyper-Elastic
# Sener=FRAME.fieldOutputs['SENER']     #Only for Visco Hyper-Elastic
Node_Tread=odb.rootAssembly.nodeSets['FOOT']
Disp_i = Disp.getSubset(region=Node_Tread)
CPress_i = C_Press.getSubset(region=Node_Tread)
CShear1_i = C_Shear1.getSubset(region=Node_Tread)
CShear2_i = C_Shear2.getSubset(region=Node_Tread)
C_NormF_i = C_NormF.getSubset(region=Node_Tread)
C_ShearF_i = C_ShearF.getSubset(region=Node_Tread)
C_NArea_i = C_NArea.getSubset(region=Node_Tread)
Time1=FRAME.frameValue

if Tempcase != 'HE':    
    NT_11=FRAME.fieldOutputs['NT11']
    Temp=FRAME.fieldOutputs['TEMP']
    
#OUT = open(Case+'_'+NAME[kk]+'.txt','w+')
# OUT = open(Case+'_F'+str(FRAME.frameId)+'.txt','w+')
OUT = open(Case+'_'+STEP+'.txt','w+')
OUT.write('#Node\t')
OUT.write('X Deformed\t')
OUT.write('Y Deformed\t')
OUT.write('Z Deformed\t')
OUT.write('CPress\t')
OUT.write('CShear1\t')
OUT.write('CShear2\t')
OUT.write('CNormF\t')
OUT.write('CShearF1\t')
OUT.write('CShearF2\t')
OUT.write('CNArea\n')
# OUT.write('CENER\n')

if Tempcase != 'HE':  
    OUT.write('SENER\t')
    OUT.write('NT11\t')
    OUT.write('TEMP\n')
# else:
#     OUT.write('SENER\n')
    
Node1=[]
for jj in xrange(len(Disp_i.values)):
    Node1=Node1+[Disp_i.values[jj].nodeLabel]
Node2=[]
for kk in xrange(len(Node_Tread.nodes[0])):
    Node2=Node2+[Node_Tread.nodes[0][kk].label]

CNORMF_sum = 0.0
for ii in xrange(len(CPress_i.values)):
    n=CPress_i.values[ii].nodeLabel

    jj=Node1.index(n)
    ux=Disp_i.values[jj].data[0]
    uy=Disp_i.values[jj].data[1]
    uz=Disp_i.values[jj].data[2]

    kk=Node2.index(n)
    x=Node_Tread.nodes[0][kk].coordinates[0]
    y=Node_Tread.nodes[0][kk].coordinates[1]
    z=Node_Tread.nodes[0][kk].coordinates[2]

    OUT.write(str(n))
    OUT.write('\t')
    OUT.write(str(x+ux))
    OUT.write('\t')
    OUT.write(str(y+uy))
    OUT.write('\t')
    OUT.write(str(z+uz))
    OUT.write('\t')
    OUT.write(str(CPress_i.values[ii].data))
    OUT.write('\t')
    OUT.write(str(CShear1_i.values[ii].data))
    OUT.write('\t')
    OUT.write(str(CShear2_i.values[ii].data))
    OUT.write('\t')
    OUT.write(str(C_NormF_i.values[ii].data[2]))
    OUT.write('\t')
    OUT.write(str(C_ShearF_i.values[ii].data[0]))
    OUT.write('\t')
    OUT.write(str(C_ShearF_i.values[ii].data[1]))
    OUT.write('\t')
    OUT.write(str(C_NArea_i.values[ii].data))
    OUT.write('\t')
    # OUT.write(str(Cener.values[ii].data))     #Only for Visco Hyper-Elastic
    # OUT.write('\t')                           #Only for Visco Hyper-Elastic
    if Tempcase != 'HE': 
        OUT.write(str(Sener.values[ii].data))
        OUT.write('\t')
        OUT.write(str(NT_11.values[ii].data))
        OUT.write('\t')
        OUT.write(str(Temp.values[ii].data))
        OUT.write('\n')
    else:
        # OUT.write(str(Sener.values[ii].data)) #Only for Visco Hyper-Elastic
        OUT.write('\n')                       #Only for Visco Hyper-Elastic
        
    CNORMF_sum = CNORMF_sum + C_NormF_i.values[ii].data[2]
OUT.close()


OUT1.write(str(FRAME.frameId))
OUT1.write('\t')
OUT1.write(str(Time1))
OUT1.write('\t')
OUT1.write(str(CNORMF_sum))
OUT1.write('\n')
print >> sys.__stdout__, Time1
print >> sys.__stdout__, CNORMF_sum
    
OUT1.close()
t1 = time.clock() - t0
elapsed_time = (t1 - t0)/60.0
report_elaspse_time = (str(elapsed_time)+' min')
print >> sys.__stdout__, report_elaspse_time