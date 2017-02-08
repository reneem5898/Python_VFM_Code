import vfmFunctions
import numpy as np
import math
import time

# Constants
rho = 1e-6 # Density
f = 60 #Hz - vibration frequency
omega = f*2*np.pi

# Directory
modelDir = '/projects/uoa00138/VFM/beam1/'

# Load data - nodes, elements and displacements
# Nodes
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
print('Loading model nodes...\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

nodeFile = modelDir + 'nodeCoords.txt'
fobj = open(nodeFile)
nLines = fobj.readlines()
fobj.close()

# Put node coordinates in a dictionary
nodes = {}
for line in nLines:
    x = line.split("\t") ## needs to be changed for different models
    nodes[int(x[0].strip())] = (float(x[1].strip()), float(x[2].strip()), float(x[3].strip()))

# Elements
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
print('Loading model elements...\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

elemFile = modelDir + 'elems.txt'
fobj = open(elemFile)
eLines = fobj.readlines()
fobj.close()

# Put element node connectivity into a dictionary
elems = {}
for line in eLines:
    x = line.split("\t") ## needs to be changed for different models
    elemNodes = [int(item.strip()) for i,item in enumerate(x) if i > 0]
    elems[int(x[0].strip())] = elemNodes

# Degrees of freedom per node
DOF = 3

# Displacements
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
print('Loading displacements...\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

uFile = modelDir + 'uComplex.txt'
fobj = open(uFile)
uLines = fobj.readlines()
fobj.close()

U = []
# Put displacements into a list
for line in uLines:
    x = line.split('\t')
    U.append(complex(float(x[0].strip()) + float(x[1].strip())*1j))

# Calculate numeric virtual field
tic = time.time()
uVF = vfmFunctions.createNumericVF(U, nodes, elems, DOF)

elapsed = time.time() - tic # print how long it took
print('Elapsed time: ' + str(elapsed) + 'seconds')

# Calculate the shear modulus
print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
print('Calculating the shear modulus...\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

tic = time.time()
fk, fg, b = vfmFunctions.vfm(U, uVF, rho, omega, DOF, nodes, elems)
elapsed = time.time() - tic
print('Elapsed time: ' + str(elapsed) + 'seconds')
print('\nfk:')
print(fk)
print('fg:')
print(fg)
print('b:')
print(b)

sComplex = b/fg
shear = sComplex.real
damp = sComplex.imag/sComplex.real

outFile = open('results.txt', 'w')
outFile.write('Shear modulus:' + str(shear) + '\n' + 'Damping Coefficient: ' + str(damp) + '\n')
outFile.close()
