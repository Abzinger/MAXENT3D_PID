# test_copy_gate

from sys import path
path.insert(0,"..")

from Chicharro_pid import pid, Chicharro_pid_Exception

import time
from math import log2

for n_X in range(10,40,10):
    for n_Y in range(10,40,10):
        for n_Z in range(10,40,10): 
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
        print("Run Chicharro_PID.pid().")
        tic = time.process_time()
        returndict = pid(pdf, cone_solver="ECOS", parallel='on', output=2)
        toc = time.process_time()

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

        msg ="""Deviation from analytical results:
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
    #^ for n_Z
#^ for n_Y
#EOF
