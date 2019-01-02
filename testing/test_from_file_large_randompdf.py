# test_form_file_large_randompdfs.py
from MAXENT3D_PID import pid, MAXENT3D_PID_Exception
import time
from random import random
from sys import argv
import pickle
import numpy as np
from collections import defaultdict

# test_from_file_randompdf.py -- part of MAXENT3D_PID (https://github.com/Abzinger/MAXENT3D_PID/)
# Usage: python3 test_from_file_large_randompdfs.py t x y z # pdfs group

# Role :   If group > 0: compute PID using MAXENT3D_PID of random pdfs stored in 
#                        the file randompdfs/randompdfs_t_x_y_z_pdfs_group.pkl
#
#          else        : compute PID using MAXENT3D_PID of random pdfs stored in 
#                        the file randompdfs/randompdfs_t_x_y_z_pdfs.pkl.


# Where:   t        is the size of the range of T;
#          x        is the size of the range of X;
#          y        is the size of the range of Y;
#          z        is the size of the range of Z;
#          pdfs     is the number of pdfs in the file;
#          group    is the number of the file;


def compute_pid(nT, nX, nY, nZ, maxiter, group, maxgroup):

    # Ecos Parameters 
    parms = dict()
    parms['max_iters'] = 100
    parms['keep_solver_object'] = True

    # Lists to store results when distributions are stored in a single file 
    Ti = []
    Err = []
    UIX = []
    UIY = []
    UIZ = []
    Npid = 0
    Npiddict = defaultdict(list)

    # counter for time 
    itime = 0
    
    # Open files to read the distribitions 
    if group > 0: 
        f = open("randompdfs/randompdf_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+"_"+str(group)+".pkl", "rb")
    elif group == 0:
        f = open("randompdfs/randompdf_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "rb") 
    #^ if reading distributions
    
    # Open files to store the results for boxploting later 
    # if group > 0:

    #     # When ditributions are stored in multiple files then
    #     # Open the temperory data which contains all the data of the previous group
    #     # Read the list
    #     # Then close the file and open it again for writing (not appending)

    #     if group == 1:

    #         # Time 
    #         g_t = open("randompdfs/boxplots/temp_boxplot_time_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "wb")
    #         data_time = []

    #         # Error 
    #         g_e = open("randompdfs/boxplots/temp_boxplot_error_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "wb")

    #         data_error = []
    #     elif group > 1 and group < maxgroup:

    #         # Time 
    #         g_t = open("randompdfs/boxplots/temp_boxplot_time_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "rb")
    #         data_time = pickle.load(g_t)
    #         g_t.close()
    #         g_t = open("randompdfs/boxplots/temp_boxplot_time_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "wb")

    #         # Error
    #         g_e = open("randompdfs/boxplots/temp_boxplot_error_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "rb")
    #         data_error = pickle.load(g_e)
    #         g_e.close()
    #         g_e = open("randompdfs/boxplots/temp_boxplot_error_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "wb")

    #     elif group == maxgroup:
    #         # Time 
    #         g_t = open("randompdfs/boxplots/temp_boxplot_time_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "rb")
    #         data_time = pickle.load(g_t)
    #         g_t.close()
    #         g_t = open("randompdfs/boxplots/data_boxplot_time.pkl", "ab")

    #         # Error
    #         g_e = open("randompdfs/boxplots/temp_boxplot_error_"+str(nT)+"_"+str(nX)+"_"+str(nY)+"_"+str(nZ)+"_"+str(maxiter)+".pkl", "rb")
    #         data_error = pickle.load(g_e)
    #         g_e.close()
    #         g_e = open("randompdfs/boxplots/data_boxplot_error.pkl", "ab")
    #     #^ if opening file of index 1 or i or max_i   
   
    # elif group == 0:
    #     # If distributions are stored in a single file
    #     # Open the main data file to append to it
    #     g_t = open("randompdfs/boxplots/data_boxplot_time.pkl", "ab")
    #     g_e = open("randompdfs/boxplots/data_boxplot_error.pkl", "ab")
    #     g_x = open("randompdfs/boxplots/data_boxplot_UIX.pkl", "ab")
    #     g_y = open("randompdfs/boxplots/data_boxplot_UIY.pkl", "ab")
    #     g_z = open("randompdfs/boxplots/data_boxplot_UIZ.pkl", "ab")
        
    #^ if storing data 

    # Starts Computing

    for iter in range(maxiter):
        print("Random PDFs   with |T| =", nT, "|X| =",nX, "|Y| =",nY, " |Z| =",nZ)
        print("______________________________________________________________________")
        if group > 0:
            print("Read pdf #", maxiter*(group - 1) + iter)
        elif group == 0:
            print("Read pdf #", iter)
        #^ if

        # read the distribution
        pdf = pickle.load(f)

        # Compute PID using MAXENT3D_PID
        print("Run Chicharro_PID.pid().")
        itic = time.time()
        returndict = pid(pdf, cone_solver="ECOS", parallel='on', output=1, **parms)       
        itoc = time.time()

        # Print PID details
        msg="""Synergistic information: {CI}
        Unique information in X: {UIX}
        Unique information in Y: {UIY}
        Unique information in Z: {UIZ}
        Unique information in X,Y: {UIXY}
        Unique information in X,Z: {UIXZ}
        Unique information in Y,Z: {UIYZ}
        Shared information: {SI}
        Primal feasibility ( min -H(S|X,Y,Z) ): {Num_err_I[0]}
        Dual feasibility ( min -H(S|X,Y,Z) ): {Num_err_I[1]}
        Duality Gap ( min -H(S|X,Y,Z) ): {Num_err_I[2]}
        Primal feasibility ( min -H(S|X,Y) ): {Num_err_12[0]}
        Dual feasibility ( min -H(S|X,Y) ): {Num_err_12[1]}
        Duality Gap ( min -H(S|X,Y) ): {Num_err_12[2]}
        Primal feasibility ( min -H(S|X,Z) ): {Num_err_13[0]}
        Dual feasibility ( min -H(S|X,Z) ): {Num_err_13[1]}
        Duality Gap ( min -H(S|X,Z) ): {Num_err_13[2]}
        Primal feasibility ( min -H(S|Y,Z) ): {Num_err_23[0]}
        Dual feasibility ( min -H(S|Y,Z) ): {Num_err_23[1]}
        Duality Gap ( min -H(S|Y,Z) ): {Num_err_23[2]}"""
        print(msg.format(**returndict))
        print("_______________________________________")
        print("Time: ",itoc-itic,"secs")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Add to total time 
        itemp = itoc - itic
        itime += itemp

        # Compute the relative negative PID: min(PID)/MI 

        # Compute MI 
        solver    = returndict['Ecos Solver Object']
        solver_II = returndict['Opt II Solver Object']
        pdf_clean = { k:v  for k,v in pdf.items() if v > 1.e-300 }
        condentropy__orig = solver.condentropy__orig(pdf_clean, output = 0)
        entropy_S     = solver.entropy_V(1, pdf_clean, output = 0)

        # Compute min(PID)/MI 
        negPID = min( returndict['SI'], returndict['CI'],
                  returndict['UIX'], returndict['UIY'], returndict['UIZ'],
                  returndict['UIXY'], returndict['UIXZ'], returndict['UIYZ'] )
        argnegPID = np.argmin([ returndict['SI'], returndict['CI'],
                                returndict['UIX'], returndict['UIY'], returndict['UIZ'],
                                returndict['UIXY'], returndict['UIXZ'], returndict['UIYZ'] ])
        rel_negPID = negPID/ (entropy_S - condentropy__orig)

        # Store the index if relative negative PID is significant 
        if rel_negPID <= -1.e-4:
            if group > 0: 
                if returndict['SI'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('SI')
                if returndict['CI'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('CI')
                if returndict['UIX'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('UIX')
                if returndict['UIY'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('UIY')
                if returndict['UIZ'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('UIZ')
                if returndict['UIXY'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('UIXY')
                if returndict['UIXZ'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('UIXZ')
                if returndict['UIYZ'] < 0:
                    Npiddict[maxiter*(group - 1) + iter].append('UIYZ')
                Npid += 1
            elif group == 0:
                if returndict['SI'] < 0:
                    Npiddict[iter].append('SI')
                if returndict['CI'] < 0:
                    Npiddict[iter].append('CI')
                if returndict['UIX'] < 0:
                    Npiddict[iter].append('UIX')
                if returndict['UIY'] < 0:
                    Npiddict[iter].append('UIY')
                if returndict['UIZ'] < 0:
                    Npiddict[iter].append('UIZ')
                if returndict['UIXY'] < 0:
                    Npiddict[iter].append('UIXY')
                if returndict['UIXZ'] < 0:
                    Npiddict[iter].append('UIXZ')
                if returndict['UIYZ'] < 0:
                    Npiddict[iter].append('UIYZ')
                Npid += 1
            #^ if
        #^ if 

        # Compute Max Error
        err_I = returndict['Num_err_I']
        err_12 = returndict['Num_err_12']
        err_13 = returndict['Num_err_13']
        err_23 = returndict['Num_err_23']
        
        err = max( err_I[0], err_I[1], err_I[2],
                   err_12[0], err_12[1], err_12[2],
                   err_13[0], err_13[1], err_13[2],
                   err_23[0], err_23[1], err_23[2] )

        # Store the results the results
        if group > 0:
            # Store Time 
            data_time.append(itoc - itic)
        
            # Store Error
            data_error.append(err)
        elif group == 0:
            # Store  time 
            Ti.append(itoc - itic)
            # Store Error 
            Err.append(err)
            UIX.append(returndict['UIX'])
            UIY.append(returndict['UIY'])
            UIZ.append(returndict['UIZ'])
    #^ for iter
    f.close()
    print("**********************************************************************")
    print("Average time: ", (itime)/maxiter, "secs")
    print("The pdfs w/ negative PID", Npiddict)
    print("# pdfs w/ negative PID", Npid)

    # Pickle the results
    # if group > 0 and group < maxgroup:
    #     print("pickling results into randompdfs/boxplots/temp_boxplot_time_", nT,"_", nX,"_", nY,"_", nZ,"_",maxiter,".pkl ...")
    #     pickle.dump(data_time, g_t)
    #     g_t.close()
        
    #     print("pickling results into randompdfs/boxplots/temp_boxplot_error_", nT,"_", nX,"_", nY,"_", nZ,"_",maxiter,".pkl ...")
    #     pickle.dump(data_error, g_e)
    #     g_e.close()
    # elif group == maxgroup:
    #     print("pickling results into randompdfs/boxplots/data_boxplot_time.pkl ...")
    #     pickle.dump(data_time, g_t)
    #     g_t.close()
    #     print("pickling results into randompdfs/boxplots/data_boxplot_error.pkl ...")
    #     pickle.dump(data_error, g_e)
    #     g_e.close()
    # elif group == 0:
    #     print("pickling results into randompdfs/boxplots/data_boxplot_time.pkl ...")
    #     pickle.dump(Ti, g_t)
    #     g_t.close()
    #     print("pickling results into randompdfs/boxplots/data_boxplot_error.pkl ...")
    #     pickle.dump(Err, g_e)
    #     g_e.close()
    #     print("pickling results into randompdfs/boxplots/data_boxplot_UIX.pkl ...")
    #     pickle.dump(UIX, g_x)
    #     g_x.close()
    #     print("pickling results into randompdfs/boxplots/data_boxplot_UIY.pkl ...")
    #     pickle.dump(UIY, g_y)
    #     g_y.close()
    #     print("pickling results into randompdfs/boxplots/data_boxplot_UIZ.pkl ...")
    #     pickle.dump(UIZ, g_z)
    #     g_z.close()
        
    #^ if

    return 0
#^ compute_pid()


def Main(sys_argv):
    print("\ntest_from_file_randompdfs.py -- part of MAXENT3D_PID (https://github.com/Abzinger/MAXENT3D_PID/)","\n")
    if len(sys_argv)!=8:
        msg="""Usage: python3 test_from_file_large_randompdfs.py t x y z Npdfs group maxgroup

Role : If group > 0 : compute PID using MAXENT3D_PID of random pdfs stored in 
                      the file randompdfs/randompdfs_t_x_y_z_pdfs_group.pkl
                      (when pdfs w/ t, x, y, z, Npdfs parameters are stored 
                       in multiple files).

       If group == 0: compute PID using MAXENT3D_PID of random pdfs stored in 
                      the file randompdfs/randompdfs_t_x_y_z_pdfs.pkl (when 
                      pdfs w/ t, x, y, z, Npdfs parameters are stored in  a 
                      single file).

Where:  t         is the size of the range of T;
        x         is the size of the range of X;     
        y         is the size of the range of Y;
        z         is the size of the range of Z;
        num_pdfs  is the number of pdfs in the file;
        group     is the number of the file;
        maxgroup  is the maximum number of files such that |T| = t, |X| = x, 
                  |Y| = y, |Z|= z, and Npdfs = num_pdfs.
        """

        print(msg)    
        exit(0)
    #^ if
    try:
        nT        = int(sys_argv[1])
        nX        = int(sys_argv[2])
        nY        = int(sys_argv[3])
        nZ        = int(sys_argv[4])
        maxiter   = int(sys_argv[5])
        group     = int(sys_argv[6])
        maxgroup  = int(sys_argv[7])
    except:
        print("I couldn't parse one of the arguments (they must all be integers)")
        exit(1)
    #^except

    if min(nT, nX, nY, nZ) < 2:
        print("All sizes of ranges must be at least 2.")
        exit(1)
    #^ if

    if maxiter < 1:
        print("# iterations must be >= 1.")
        exit(1)
    #^ if
    if group < 0:
        print("index of a file is positive")
        exit(1)
    #^ if

    if group > maxgroup:
        print(" index of a file is smaller than the maximum index")
    #^ if
    
    # Compute PID
    compute_pid(nT, nX, nY, nZ, maxiter, group, maxgroup)
    

    return 0 
#^ Main()


# Run it!
Main(argv)

# EOF
