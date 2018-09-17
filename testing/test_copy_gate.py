# test_copy_gate.py

from sys import path
path.insert(0,"..")

from Chicharro_pid import pid, Chicharro_pid_Exception

import time
import pickle
from math import log2
from sys import argv 

##########################################################################################
# test_copy_gate.py -- part of Chicharro_PID (https://github.com/Abzinger/Chicharro_PID/)
#
# Usage: python3 test_copy_gate.py l_x u_x step_x l_y u_y step_y l_z u_z step_z
#
# Role : compute the PID using Chicharro_PID of Copy gates which is
#        Copy( X ,Y ,Z ) = (X,Y,Z) for different sizes of X, Y, and  Z;
#
# Where: |X| in range(l_x,u_x,step_x);
#        |Y| in range(l_y,u_y,step_y);
#        |Z| in range(l_z,u_z,step_z).
##########################################################################################

def compute_copy_pid(l_X, u_X, step_X, l_Y, u_Y, step_Y, l_Z, u_Z, step_Z):

    # Lists to store Time for ploting boxplots
    Ti = []

    # List to store detected negative PIDs
    Npid = []

    # Files to save Ti and Npid for boxplotting 
    time_file  = open("copygate_time.pkl", 'ab')
    Npid_file  = open("copygate_negative_pid.pkl", 'ab')

    # Solver parameters 
    parms = dict()
    parms['max_iters'] = 100
    parms['keep_solver_object'] = True

    # Main loop over |X| ,|Y| , and |Z|
    for n_X in range(l_X,u_X,step_X):
        for n_Y in range(l_Y,u_Y,step_Y):
            for n_Z in range(l_Z,u_Z,step_Z):

                # Create PDF
                print("______________________________________________________________________")
                print("COPY   with |X| =",n_X,", |Y| =",n_Y,", |Z| =",n_Z,":")
                print("Create pdf.")
                pdf = dict()
                for x in range(n_X):
                    for y in range(n_Y):
                        for z in range(n_Z):
                            s = (x,y,z)
                            pdf[ (s,x,y,z) ] = 1./(n_X*n_Y*n_Z)
                        #^ for z
                    #^ for y
                #^ for x

                # Compute PID 
                print("Run Chicharro_PID.pid().")
                tic = time.time()
                returndict = pid(pdf, cone_solver="ECOS", parallel='on', output=2, **parms)
                toc = time.time()

                # Compute deviations from the analytical results
                returndictdev = dict()
                returndictdev['CI']  = 100*abs(returndict['CI'] - 0.)
                returndictdev['SI']  = 100*abs(returndict['SI'] - 0.)
                returndictdev['UIXY']  = 100*abs(returndict['UIXY'] - 0.)
                returndictdev['UIXZ']  = 100*abs(returndict['UIXZ'] - 0.)
                returndictdev['UIYZ']  = 100*abs(returndict['UIYZ'] - 0.)
                returndictdev['UIX'] = 100*abs(returndict['UIX'] - log2(n_X))
                returndictdev['UIY'] = 100*abs(returndict['UIY'] - log2(n_Y))
                returndictdev['UIZ'] = 100*abs(returndict['UIZ'] - log2(n_Z))
                returndictdev['Num_err_I'] = returndict['Num_err_I']
                returndictdev['Num_err_12'] = returndict['Num_err_12']
                returndictdev['Num_err_13'] = returndict['Num_err_13']
                returndictdev['Num_err_23'] = returndict['Num_err_23']
                returndictdev['Time'] = toc-tic

                # Print the result
                msg ="""\nDeviation from analytical results:
                Synergistic information: {CI} %
                Unique information in X: {UIX} %
                Unique information in Y: {UIY} %
                Unique information in Z: {UIZ} %
                Unique information in X,Y: {UIXY} %
                Unique information in X,Z: {UIXZ} %
                Unique information in Y,Z: {UIYZ} %
                Shared information: {SI} %
                +++++++++++++++++++++++++++++++++++
            
                Optimization Quality:
                Primal feasibility ( min H(S|X,Y,Z) ): {Num_err_I[0]}
                Dual feasibility ( min H(S|X,Y,Z) ): {Num_err_I[1]}
                Duality Gap ( min H(S|X,Y,Z) ): {Num_err_I[2]}
                Primal feasibility ( min H(S|X,Y) ): {Num_err_12[0]}
                Dual feasibility ( min H(S|X,Y) ): {Num_err_12[1]}
                Duality Gap ( min H(S|X,Y) ): {Num_err_12[2]}
                Primal feasibility ( min H(S|X,Z) ): {Num_err_13[0]}
                Dual feasibility ( min H(S|X,Z) ): {Num_err_13[1]}
                Duality Gap ( min H(S|X,Z) ): {Num_err_13[2]}
                Primal feasibility ( min H(S|Y,Z) ): {Num_err_23[0]}
                Dual feasibility ( min H(S|Y,Z) ): {Num_err_23[1]}
                Duality Gap ( min H(S|Y,Z) ): {Num_err_23[2]}
                +++++++++++++++++++++++++++++++++++++++++++
            
                Time: {Time} sec"""
                print(msg.format(**returndictdev))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                # Store Time for later usage
                Ti.append(toc - tic)

                
                # Compute MI(S; X,Y,Z)
                solver    = returndict['Ecos Solver Object']
                solver_II = returndict['Opt II Solver Object']
                pdf_clean = { k:v  for k,v in pdf.items() if v > 1.e-300 }
                condentropy__orig = solver.condentropy__orig(pdf_clean)
                entropy_S     = solver_II.entropy_S(pdf_clean)

                # Compute relative negativity
                neg = min( returndict['SI'], returndict['CI'],
                  returndict['UIX'], returndict['UIY'], returndict['UIZ'],
                  returndict['UIXY'], returndict['UIXZ'], returndict['UIYZ'] )
                
                rel_neg = neg/ (entropy_S - condentropy__orig)

                # Store the (|X|,|Y|,|Z|) when rel_neg is significant 
                if rel_neg <= -1.e-4:
                    Npid.append((nX,nY,nZ))
                #^ if rel_neg
            #^ for n_Z
        #^ for n_Y
    #^ for n_X

    # Store into times and error file to create boxplots later
    pickle.dump(Ti, time_file)
    pickle.dump(Npid, Npid_file)
    time_file.close()
    Npid_file.close()

    return Npid
#^ compute_copy_pid()

def Main(sys_argv):
    print("test_copy_gate.py -- part of Chicharro_PID (https://github.com/Abzinger/Chicharro_PID/)")
    if len(argv) != 10:
        msg ="""Usage: python3 test_copy_gate.py l_x u_x step_x l_y u_y step_y l_z u_z step_z

Role : compute the PID using Chicharro_PID of Copy gates 
       which is Copy( X ,Y ,Z ) = (X,Y,Z) 
       for different sizes of X, Y, and  Z

Where: |X| in range(l_x,u_x,step_x);
       |Y| in range(l_y,u_y,step_y);
       |Z| in range(l_z,u_z,step_z).
"""
        print(msg)
        exit(0)
    #^ if

    # Checks if all inputs are integers
    try:
        l_X    = int(argv[1])
        u_X    = int(argv[2])
        step_X = int(argv[3])
        l_Y    = int(argv[4])
        u_Y    = int(argv[5])
        step_Y = int(argv[6])
        l_Z    = int(argv[7])
        u_Z    = int(argv[8])
        step_Z = int(argv[9])
    except:
        print("I couldn't parse one of the arguments (they must all be integers)")
        exit(1)
    #^except

    # Checks if |X| > 1, |Y| > 1, |Z| > 1
    if min(l_X, u_X, l_Y, u_Y, l_Z, u_Z) < 2:
        print("Error: All sizes of ranges must be at least 2.")
        exit(1)
    #^ if range < 2

    # Checks if: l_X < u_X, l_Y < u_Y, l_Z < u_Z   
    if l_X >= u_X:
        print("Error: l_X must be smaller than  u_X")
        exit(1)
    elif l_Y >= u_Y:
        print("Error: l_Y must be smaller than u_Y")
        exit(1)
    elif l_Z >= u_Z:
        print("Error: l_Z must be smaller than u_Z")
        exit(1)
    #^ if lower > upper 

    # Checks if step_X < u_X, step_Y < u_Y, step_Z < u_Z,
    if step_X >= u_X:
        print(" Warning |X| =", l_X," since step_X:", step_X,"is larger than or equal u_X:", u_X)
    elif step_Y >= u_Y:
        print(" Warning |Y| =", l_Y," since step_Y:", step_Y,"is larger than or equal u_Y:", u_Y)
    elif step_Z >= u_Z:
        print(" Warning |Z| =", l_Z," since step_Z:", step_Z," is larger than u_Z:", u_Z)
    #^ if step > upper

    # Compute PID
    compute_copy_pid(l_X, u_X, step_X, l_Y, u_Y, step_Y, l_Z, u_Z, step_Z)

    return 0
#^ Main()
    
Main(argv) 
#EOF
