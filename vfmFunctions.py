def createNumericVF(U, nodes, elems, DOF):
    
##############################################################################
# This function calculates numeric virtual displacement field following the
# methods outlined in Connesson et al. for a harmonic displacement field
# The following conditions are specified:
# 1. fk = 0 (Ak)
# 2. fg = 1 (Ag)
# 3. sum(uVFx) = 0, sum(uVFy) = 0, sum(uVFz) = 0 (Arb)
# 4. uVF(boundaries) = 0 (Acl)
# 5. Noise minimisation (H)
#
# Inputs: 1) u - MRE displacement field
#         2) nodes - list of node numbers and nodal coordinates (numNodes x 3)
#         3) elems - (numElems x 8) - 8 node numbers that make up the
#         element
#         4) DOF - number of degrees of freedom for each node
#
# Outputs: 1) uVF - special and optimised virtual displacement field
#          2 - 6) Components of condition/specialisation matrix
#       
#
# Written by: Renee Miller
# Date: 26 August 2016
############################################################################

    import math
    import scipy.sparse
    import scipy.sparse.linalg
    import numpy as np

    ###### Step 1: create boundary constraints
    # Get list of boundary nodes in the model (or subzone)
    boundaryNodes = getBoundaryNodes(elems)

    # Use list of boundary nodes to construct the boundary constraint matrices
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    print('Constructing boundary constraint...\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
    Ab, RHS_Ab = boundaryConstraint(boundaryNodes, nodes, DOF)
    print(Ab)

    ###### Step 2: create rigid body constraint
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    print('Constructing rigid body constraint (min Fk) ...\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
    Arb = rigidBodyConstraint(nodes, DOF)
    print(Arb)

    ###### Step 3: create Ak, Ag and H constraints

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    print('Constructing Ak, Ag and H constraints...\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

    # Initialise Ak and Ag constraint vectors
    Ak = np.zeros((1,len(U)), dtype=complex)
    Ag = np.zeros((1,len(U)), dtype=complex)

    # Initialise H matrix
    H = np.zeros((len(nodes)*DOF, len(nodes)*DOF), dtype=complex)

    # Loop through elements
    elList = [1]
    for i in sorted(elems.keys()):
#    for i in elList:
        elemNodes = elems[i]
        # Calculate componenents of constraint matrices
        ak, ag, h, nodeIdcs = elemConstraints_H_Ak_Ag(elemNodes, nodes, U, DOF)
                
        # Add constraints
        for i,item in enumerate(nodeIdcs):
            Ak[0,item] = Ak[0,item] + complex(ak[0,i])
            Ag[0,item] = Ag[0,item] + complex(ag[0,i])

        # Assemble H matrix
        for i,item1 in enumerate(nodeIdcs):
            for j,item2 in enumerate(nodeIdcs):
                H[item1, item2] = H[item1, item2] + complex(h[i,j])

    print('Ak:')
    print(Ak)
    print('Ag:')
    print(Ag)

    ###### Step 4: compile the constraint matrices and solve
    # Compile A matrix
    A = scipy.sparse.vstack([scipy.sparse.csr_matrix(m) for m in [Ab, Arb, Ak, Ag]])
    #A = scipy.sparse.csr_matrix(A)

    # Compile LHS
    z1 = scipy.sparse.csr_matrix((A.shape[0], A.shape[0]))
    LHS_t = scipy.sparse.hstack([scipy.sparse.csr_matrix(H), A.transpose()])
    LHS_b = scipy.sparse.hstack([A, z1])
    LHS_csr = scipy.sparse.vstack([LHS_t, LHS_b])

    # Compile RHS
    hShape = H.shape
    constraint = np.zeros((4,1), dtype=complex)
    fgConstraint = np.matrix([1.0 + 0j])
    Zg = np.concatenate((RHS_Ab, constraint, fgConstraint), axis=0)
    z2 = np.zeros((hShape[0],1))
    RHS = np.concatenate((z2, Zg), axis=0)
    print('Zg:')
    print(Zg)
    print('RHS:')

    # Convert matrices to sparse matrices
    #LHS_csr = scipy.sparse.csr_matrix(LHS)
    RHS_csr = scipy.sparse.csr_matrix(RHS)
    print(RHS)

    # Solve for virtual displacement field
    x = scipy.sparse.linalg.spsolve(LHS_csr, RHS_csr)
    print(x.shape)

    # Return just uVF
    uVF = x[:hShape[0]]
    print('uVF:')
    print(uVF)
    uVF = np.asarray(uVF)
    uVF = uVF.tolist()
    print(uVF)

    return uVF

#################################################################################################################

def elemConstraints_H_Ak_Ag(elemNodes, nodes, U, DOF):

    import numpy as np
    import math

    # Get vector of x and y coordinates
    nodeNums = sorted(list(nodes.keys()))
    X = []
    Y = []
    Z = []
    for node in elemNodes:
        X.append(nodes[node][0])
        Y.append(nodes[node][1])
        Z.append(nodes[node][2])

    # Calculate delX, delY and delZ - average length of sides of element
    delX, delY, delZ = calcHexSides(X, Y, Z)

    # Get node indices for extracting nodal displacements in current element
    nodeIdcs = [(((nodeNums.index(node)+1)*DOF)-(DOF-d)) for node in elemNodes for d in range(0,DOF)]

    # Get displacemnets at element nodes
    Ue = [U[i] for i in nodeIdcs]

    # Get local coordiinate of gauss point - only using 1 gauss point currently
    zeta = 0 # gauss point
    w = 2 # gauss weight
    
    # Coordinates of gauss point
    o = zeta
    n = zeta
    m = zeta

    # Evaluate the derivative of the shape function at m, n, o
    DN = 0.125*np.matrix([[-1*(1-n)*(1-o), (1-n)*(1-o), (1+n)*(1-o), -1*(1+n)*(1-o), -1*(1-n)*(1+o), (1-n)*(1+o), (1+n)*(1+o), -1*(1+n)*(1+o)], [-1*(1-m)*(1-o), -1*(1+m)*(1-o), (1+m)*(1-o), (1-m)*(1-o), -1*(1-m)*(1+o), -1*(1+m)*(1+o), (1+m)*(1+o), (1-m)*(1+o)], [-1*(1-m)*(1-n), -1*(1+m)*(1-n), -1*(1+m)*(1+n), -1*(1-m)*(1+n), (1-m)*(1-n), (1+m)*(1-n), (1+m)*(1+n), (1-m)*(1+n)]])

    # Convert the coordinate vectors to a matrix
    coords = [[X[i], Y[i], Z[i]] for i in range(len(X))]
    coordsMat = np.asmatrix(coords)

    # Calculate jacobian
    jac = DN*coordsMat

    # Multiply inverse of jacobian times the derivative of the shape function
    dNdXYZ = np.linalg.solve(jac,DN)

    B = np.zeros((6,1)) # Get rid of this afterwards
    # Calculate B matrix (strain matrix)
    for x in range(0,len(elemNodes)):
        Bi = np.matrix([[dNdXYZ[0,x], 0, 0], [0, dNdXYZ[1,x], 0], [0, 0, dNdXYZ[2,x]], [dNdXYZ[1,x], dNdXYZ[0,x], 0], [dNdXYZ[2,x], 0, dNdXYZ[0,x]], [0, dNdXYZ[2,x], dNdXYZ[1,x]]])
        B = np.concatenate((B, Bi), axis=1)

    B = B[:,1:] # Get rid of first column
    B = np.asmatrix(B) # Convert to matrix

    # Calculate strain: B*U
    eV = np.dot(B,Ue)
    
    # Convert strain to square matrix
    e = np.matrix([[eV[0,0], 0.5*eV[0,3], 0.5*eV[0,4]], [0.5*eV[0,3], eV[0,1], 0.5*eV[0,5]], [0.5*eV[0,4], 0.5*eV[0,5], eV[0,2]]])

    # Ak term for current element - row vector
    # fk = ak * UeVF = 0
    B_tr = B[[0,1,2],:] # First three rows of B matrix - used for calculating the trace function
    ak = np.trace(e)*np.sum(B_tr,axis=0)*np.linalg.det(jac)
    
    # Ag term for current element - row vector
    # fg = ag * UeVF = 1
    tmp = eV[0,0]*B[0,:] + eV[0,1]*B[1,:] + eV[0,2]*B[2,:] + 2*0.5*eV[0,3]*0.5*B[3,:] + 2*0.5*eV[0,4]*0.5*B[4,:] +2*0.5*eV[0,5]*0.5*B[5,:]
    ag = 2*(tmp - (1/3)*np.trace(e)*np.sum(B_tr,axis=0))*np.linalg.det(jac)

    # Calculate Hg matrix for current element
    t = (1/3) * np.sum(B_tr,axis=0)
    
    # Calculate each component of Hg
    h1 = (4/math.pow(delX,2))*np.transpose(B[0,:] - t)*(B[0,:] - t)
    h2 = (4/math.pow(delY,2))*np.transpose(B[1,:] - t)*(B[1,:] - t)
    h3 = (4/math.pow(delZ,2))*np.transpose(B[2,:] - t)*(B[2,:] - t)
    h12 = (1/math.pow(delX,2) + 1/math.pow(delY,2)) * np.transpose(0.5*B[3,:]) * (0.5*B[3,:])
    h13 = (1/math.pow(delX,2) + 1/math.pow(delZ,2)) * np.transpose(0.5*B[4,:]) * (0.5*B[4,:])
    h23 = (1/math.pow(delZ,2) + 1/math.pow(delY,2)) * np.transpose(0.5*B[5,:]) * (0.5*B[5,:])

    h = math.pow(np.linalg.det(jac),2)*(h1 + h2 + h3 + h12 + h13 + h23)

    # Weight the components by w
    ak = w*w*w*ak
    ag = w*w*w*ag
    h = w*w*w*h

    return ak, ag, h, nodeIdcs 

#################################################################################################################

def calcHexSides(X, Y, Z):

    import math
    import numpy as np

    # Set connectivity: hard-coded
    conn1 = [1, 2, 3, 4, 1, 3, 5, 7, 1, 2, 5, 6]
    conn2 = [5, 6, 7, 8, 2, 4, 6, 8, 4, 3, 8, 7]

    # calculate length of each side
    lensideX = []
    lensideY = []
    lensideZ = []

    # loop through connectivity of element and calculate length of each side
    D = []
    for i in range(0,len(conn1)):
        x1 = X[conn1[i]-1] # Subtract one because connectivity info was written with indexing starting from 1
        x2 = X[conn2[i]-1]
        y1 = Y[conn1[i]-1]
        y2 = Y[conn2[i]-1]
        z1 = Z[conn1[i]-1]
        z2 = Z[conn2[i]-1]

        d = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2) + math.pow((z1-z2),2))
        D.append(d)

    dX = np.mean(D[0:3])
    dY = np.mean(D[4:7])
    dZ = np.mean(D[8:11])
    
    return dX, dX, dZ
    
#################################################################################################################

def getBoundaryNodes(elems):
    
    import numpy as np

    # Numer of nodes in each element
    nodesInEachElem = len(elems[1])

    # Get list of node numbers in all elements - used to count occurrences of each node
    nodesInElems = []
    for i in sorted(elems.keys()):
        for n in elems[i]:
            nodesInElems.append(n)
    nodesInElems.sort()
    
    # Get just list of unique node numbers
    nodesUnique = list(set(nodesInElems))

    # Make list of boundary nodes - those which have occurrences < # nodes in each element
    boundaryNodes = []
    for node in nodesUnique:
        count = nodesInElems.count(node)
        if count < nodesInEachElem:
            boundaryNodes.append(node)
    
    return boundaryNodes

##################################################################################################################

def boundaryConstraint(boundaryNodes, nodes, DOF):

    import numpy as np

    # Create matrix of zeros - boundary constraint
    Ab = np.zeros((len(boundaryNodes)*DOF,len(nodes)*DOF))
    Ab = np.asmatrix(Ab)
    
    c = 0 # Set counter
    
    # Get list of node numbers
    nodeNums = sorted(list(nodes.keys()))

    # Loop through boundary nodes
    for bn in boundaryNodes:
        # Loop through degrees of freedom
        for d in range(0,DOF):

            # Node index in global matrices
            nodeIDX = nodeNums.index(bn)
            # Index in list of all dofs
            dofIDX = (nodeIDX+1)*DOF - (DOF - d)
            
            # Set boundary constraint matrix to 1 at boundary node for each DOF
            Ab[c,dofIDX] = 1

            c = c + 1 # counter


    # Create RHS boundary constraint - displacement at these DOFS = 0
    RHS_Ab = np.zeros((c,1), dtype=complex)
    RHS_Ab = np.asmatrix(RHS_Ab) # Convert to matrix    

    return Ab, RHS_Ab

###################################################################################################################

def rigidBodyConstraint(nodes, DOF):
    
    import numpy as np
    
    # Initialist constraint matrix
    Arb = np.zeros((DOF, len(nodes)*DOF))
    
    # Construct matrix
    for d in range(0,DOF):
        Arb[d,d::3] += 1

    # Convert to matrix 
    Arb = np.asmatrix(Arb)
    
    return Arb

###################################################################################################################

def vfm(u, uVF, rho, omega, DOF, nodes, elems):

    import numpy as np
    import math

    # Get local coordiinate of gauss point - only using 1 gauss point currently
    zeta = 0 # gauss point
    w = 2 # gauss weight

    # Coordinates of gauss point
    o = zeta
    n = zeta
    m = zeta

    # Initialise variables
    FK = 0
    FG = 0
    ACC = 0

    # Shape functions
    ShapeFuns = np.array([(x*y*z/8) for z in [1-o,1+o] for y in [1-n,1+n] for x in [1-m,1+m]])
    print('Shape Functions:')
    print(ShapeFuns)

    # Derivative of shape functions
    DN = 0.125 * np.matrix([[-1*(1-n)*(1-o), (1-n)*(1-o), (1+n)*(1-o), -1*(1+n)*(1-o), -1*(1-n)*(1+o), (1-n)*(1+o), (1+n)*(1+o), -1*(1+n)*(1+o)], [-1*(1-m)*(1-o), -1*(1+m)*(1-o), (1+m)*(1-o), (1-m)*(1-o), -1*(1-m)*(1+o), -1*(1+m)*(1+o), (1+m)*(1+o), (1-m)*(1+o)], [-1*(1-m)*(1-n), -1*(1+m)*(1-n), -1*(1+m)*(1+n), -1*(1-m)*(1+n), (1-m)*(1-n), (1+m)*(1-n), (1+m)*(1+n), (1-m)*(1+n)]]) 

    # Loop through elements
    for i in sorted(elems.keys()):
        
        elemNodes = elems[i]

        # Get vector of x and y coordinates
        nodeNums = sorted(list(nodes.keys()))
        X = []
        Y = []
        Z = []
        for node in elemNodes:
            X.append(nodes[node][0])
            Y.append(nodes[node][1])
            Z.append(nodes[node][2])

        # Get node indices for extracting nodal displacements in current element
        nodeIdcs = [(((nodeNums.index(node)+1)*DOF)-(DOF-d)) for node in elemNodes for d in range(0,DOF)]

        # Get displacemnets at element nodes
        Ue = [u[i] for i in nodeIdcs]
        UeVF = [uVF[i] for i in nodeIdcs]

        # Calculate the jacobian
        coords = [[X[i], Y[i], Z[i]] for i in range(len(X))]
        coordsMat = np.asmatrix(coords)
        jac = DN * coordsMat

         # Multiply inverse of jacobian times the derivative of the shape function
        dNdXYZ = np.linalg.solve(jac,DN)

        B = np.zeros((6,1)) # Get rid of this afterwards
        # Calculate B matrix (strain matrix)
        for x in range(0,len(elemNodes)):
            Bi = np.matrix([[dNdXYZ[0,x], 0, 0], [0, dNdXYZ[1,x], 0], [0, 0, dNdXYZ[2,x]], [dNdXYZ[1,x], dNdXYZ[0,x], 0], [dNdXYZ[2,x], 0, dNdXYZ[0,x]], [0, dNdXYZ[2,x], dNdXYZ[1,x]]])
            B = np.concatenate((B, Bi), axis=1)

        B = B[:,1:] # Get rid of first column
        B = np.asmatrix(B) # Convert to matrix

        # Calculate strain: B*U
        eV = np.dot(B,Ue)
        eV_VF = np.dot(B,UeVF)

        # Convert strain to square matrix
        e = np.matrix([[eV[0,0], 0.5*eV[0,3], 0.5*eV[0,4]], [0.5*eV[0,3], eV[0,1], 0.5*eV[0,5]], [0.5*eV[0,4], 0.5*eV[0,5], eV[0,2]]])
        eVF = np.matrix([[eV_VF[0,0], 0.5*eV_VF[0,3], 0.5*eV_VF[0,4]], [0.5*eV_VF[0,3], eV_VF[0,1], 0.5*eV_VF[0,5]], [0.5*eV_VF[0,4], 0.5*eV_VF[0,5], eV_VF[0,2]]])

        # Calculate fk
        fk = np.trace(e)*np.trace(eVF)*np.linalg.det(jac)

        # Calculate fg
        fg = 2*(np.trace(e*np.transpose(eVF)) - (1/3)*np.trace(e)*np.trace(eVF))*np.linalg.det(jac)

        # Compute RHS of equation
        r = np.identity(DOF)
        N = np.zeros((DOF,1))
        for sf in range(0,len(ShapeFuns)):
            N = np.concatenate((N,r*ShapeFuns[sf]), axis=1)
        N = N[:,1:]
       
        # RHS = Integral(rho*omega^2*u*uVF*dv)
        sh = np.dot(np.transpose(N),N)
        sh = np.diag(np.sum(sh, axis=0))

        # Convert components to matrices
        sh = np.asmatrix(sh) # 24 x 24 matrix
        UeVF = np.asmatrix(UeVF) # row vector
        print('UeVF:')
        print(UeVF)
        Ue = np.transpose(np.asmatrix(Ue)) # column vector

        acc = rho*(math.pow(omega,2))*UeVF*sh*Ue*np.linalg.det(jac)
        print('acc:')
        print(acc)

        # Weight components
        fk = w*w*w*fk
        fg = w*w*w*fg
        b = w*w*w*acc

        FK = FK + fk
        FG = FG + fg
        ACC = ACC + b

    return FK, FG, ACC

