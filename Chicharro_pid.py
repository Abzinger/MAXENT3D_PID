# Chicharro_pid.py -- Python module
#
# Chicharro_pid: Chicharro trivariate Partial Information Decomposition
# https://github.com/Abzinger/Chicharro_pid 
# (c) Abdullah Makkeh, Dirk Oliver Theis
# Permission to use and modify with proper attribution
# (Apache License version 2.0)
#
# Information about the algorithm, documentation, and examples are here:
# @Article{?????????????,
#          author =       {Makkeh, Abdullah and Theis, Dirk Oliver and Vicente, Raul and Chicharro, Daniel},
#          title =        {????????},
#          journal =      {????????},
#          year =         ????,
#          volume =    {??},
#          number =    {?},
#          pages =     {???}
# }
# Please cite this paper when you use this software (cf. README.md)
##############################################################################################################
import TRIVARIATE_SYN
import TRIVARIATE_UNQ

import multiprocessing as mp
from multiprocessing import Pool
import ecos
from scipy import sparse
import numpy as np
from numpy import linalg as LA
import math
from collections import defaultdict
import time

log = math.log2
ln  = math.log

def r_vidx(i):
    return 3*i
def p_vidx(i):
    return 3*i+1
def q_vidx(i):
    return 3*i+2

class Chicharro_pid_Exception(Exception):
    pass


class Solve_w_ECOS():
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0

    def __init__(self, marg_sx, marg_sy, marg_sz, marg_xy, marg_xz, marg_yz):
        # (c) Abdullah Makkeh, Dirk Oliver Theis
        # Permission to use and modify under Apache License version 2.0

        # ECOS parameters
        self.ecos_kwargs   = dict()
        self.verbose       = False

        # Probability density funciton data
        self.b_sx         = dict(marg_sx)
        self.b_sy         = dict(marg_sy)
        self.b_sz         = dict(marg_sz)

        self.b_xy         = dict(marg_xy)
        self.b_xz         = dict(marg_xz)
        self.b_yz         = dict(marg_yz)

        self.S            =set([ s for s,x in self.b_sx.keys() ]
                               + [ s for s,y in self.b_sy.keys() ]
                               + [ s for s,z in self.b_sz.keys() ])
        self.X            = set( [ x  for s,x in self.b_sx.keys() ] )
        self.Y            = set( [ y  for s,y in self.b_sy.keys() ] )
        self.Z            = set( [ z  for s,z in self.b_sz.keys() ] )
        self.idx_of_quad  = dict()
        self.quad_of_idx  = []

        # Do stuff:
        for s in self.S:
            for x in self.X:
                if (s,x) in self.b_sx.keys():
                    for y in self.Y:
                        if (s,y) in self.b_sy.keys():
                            for z in self.Z:
                                if (s,z) in self.b_sz.keys():
                                    self.idx_of_quad[ (s,x,y,z) ] = len( self.quad_of_idx )
                                    self.quad_of_idx.append( (s,x,y,z) )
                                #^ if
                            #^ for z
                        #^ if
                    #^ for y
                #^ if
            #^ for x
        #^ for s
    #^ init()

    def condentropy__orig(self,pdf, output):
        itic = time.process_time()
        mysum = 0.
        marg_xyz = defaultdict(lambda: 0.)
        for sxyz, i in pdf.items():
            s,x,y,z = sxyz
            marg_xyz[x,y,z] += pdf[(s,x,y,z)]
        #^ for
        
        for sxyz, i in pdf.items():
            s,x,y,z = sxyz
            p = pdf[(s,x,y,z)]
            mysum -= p*log(p/marg_xyz[x,y,z])
        #^ for
        itoc = time.process_time()
        if output == 2: print("Time to compute H(S|XYZ) of the input pdf:", itoc - itic, "secs")
        return mysum
    #^ condentropy__orig()

    # Compute H(S), H(X), H(Y), or H(Z)
    def entropy_V(self, V, pdf, output):
        marg_V = defaultdict(lambda: 0.)
        num_V  = defaultdict(lambda: 0.) 
        mysum = 0.
        if V == 1:
            # H(S)
            itic = time.process_time()
            for sxyz,r in pdf.items():
                s,x,y,z = sxyz
                if r > 0:
                    marg_V[s] += r
                    num_V[s] = 1
                #^ if 
            #^ for

            for s in num_V.keys():
                if marg_V[s] > 0: mysum -= marg_V[s]*log( marg_V[s] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("Time to compute H(S):", itoc - itic, "secs")
            return mysum

        elif V == 2:
            # H(X)
            itic = time.process_time()
            for sxyz,r in pdf.items():
                s,x,y,z = sxyz
                if r > 0:
                    marg_V[x] += r
                    num_V[x] = 1
                #^ if 
            #^ for

            for x in num_V.keys():
                mysum -= marg_V[x]*log( marg_V[x] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("Time to compute H(X):", itoc - itic, "secs")
            return mysum

        elif V == 3:
            # H(Y)
            itic = time.process_time()
            for sxyz,r in pdf.items():
                s,x,y,z = sxyz
                if r > 0:
                    marg_V[y] += r
                    num_V[y] = 1
                #^ if 
            #^ for

            for y in num_V.keys():
                mysum -= marg_V[y]*log( marg_V[y] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("Time to compute H(Y):", itoc - itic, "secs")
            return mysum

        elif V == 4:
            # H(Z)
            itic = time.process_time()
            for sxyz,r in pdf.items():
                s,x,y,z = sxyz
                if r > 0:
                    marg_V[z] += r
                    num_V[z] = 1
                #^ if 
            #^ for

            for z in num_V.keys():
                mysum -= marg_V[z]*log( marg_V[z] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("Time to compute H(Z):", itoc - itic, "secs")
            return mysum
        else:
            print("The argument V takes the values: 1, 2, 3, or 4")
            exit(1)
    #^ entropy()

#^ class Solve_w_ECOS


# Subclass to compute Synergistic Information
class Opt_I(Solve_w_ECOS):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    def create_model(self, output):
        return TRIVARIATE_SYN.create_model(self, output)
    
    def solve(self, c, G, h, dims, A, b, output):
        return TRIVARIATE_SYN.solve(self, c, G, h, dims, A, b, output)
    
    def dual_value(self, sol_lambda, b):
        return TRIVARIATE_SYN.dual_value(self, sol_lambda, b)

    def check_feasibility(self, sol_rpq, sol_lambda, output):
        return TRIVARIATE_SYN.check_feasibility(self, sol_rpq, sol_lambda, output)
    
    def condentropy(self, sol_rpq, output):
        return TRIVARIATE_SYN.condentropy(self, sol_rpq, output)
#^ subclass Opt_I

# Subclass to compute Unique Information 
class Opt_II(Solve_w_ECOS):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    def initialization(self, which_sources):
        return TRIVARIATE_UNQ.initialization(self, which_sources)

    def sq_vidx(self, i, which_sources):
        return TRIVARIATE_UNQ.sq_vidx(self, i, which_sources)

    def marginals(self, which_sources, sol_rpq, output):
        return TRIVARIATE_UNQ.marginals(self, which_sources,sol_rpq, output)

    def create_model(self, which_sources, output):
        return TRIVARIATE_UNQ.create_model(self, which_sources, output)
    
    def solve(self, c, G, h, dims, A, b, output):
        return TRIVARIATE_UNQ.solve(self, c, G, h, dims, A, b, output)
    
    def dual_value(self, sol_lambda, b):
        return TRIVARIATE_UNQ.dual_value(self, sol_lambda, b)

    def check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu, output):
        return TRIVARIATE_UNQ.check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu, output)
    
    def condentropy_2vars(self, which_sources,sol_rpq,output,marg_XY,marg_XZ,marg_YZ,marg_SXY,marg_SXZ,marg_SYZ):
        return TRIVARIATE_UNQ.condentropy_2vars(self, which_sources,sol_rpq,output,marg_XY,marg_XZ,marg_YZ,marg_SXY,marg_SXZ,marg_SYZ)

    def condentropy_1var(self, which_sources, sol_rpq, marg_SX, marg_SY,marg_SZ,marg_X,marg_Y,marg_Z):
        return TRIVARIATE_UNQ.condentropy_1var(self, which_sources,sol_rpq,marg_SX,marg_SY,marg_SZ,marg_X,marg_Y,marg_Z)

    def entropy_S(self, pdf, sol_rpq, which_sources, output):
        return TRIVARIATE_UNQ.entropy_S(self, pdf, sol_rpq, which_sources, output)
#^ subclass Opt_II

# Compute Marginals

def marginal_sx(p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (s,x) in marg.keys():    marg[(s,x)] += r
        else:                       marg[(s,x)] =  r
    return marg
#^ marginal_sx()

def marginal_sy(p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (s,y) in marg.keys():   marg[(s,y)] += r
        else:                      marg[(s,y)] =  r
    return marg
#^ marginal_sy()

def marginal_sz(p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (s,z) in marg.keys():   marg[(s,z)] += r
        else:                      marg[(s,z)] =  r
    return marg
#^ marginal_sz()

def marginal_xy(p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (x,y) in marg.keys():   marg[(x,y)] += r
        else:                      marg[(x,y)] =  r
    return marg
#^ marginal_xy()

def marginal_xz(p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (x,z) in marg.keys():   marg[(x,z)] += r
        else:                      marg[(x,z)] =  r
    return marg
#^ marginal_xz()

def marginal_yz(p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (y,z) in marg.keys():   marg[(y,z)] += r
        else:                      marg[(y,z)] =  r
    return marg
#^ marginal_yz()


# Compute Conditional Entopy of the form H(W|T)
def condent_V(V, p, output = 0):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0

    # Initialization
    marg_V = defaultdict(lambda: 0.)
    mysum = 0.
    
    # Compute H(S|X)
    if V == 1:
        itic = time.process_time()
        b_sx = marginal_sx(p)

        # Get P(*,x,*,*)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if r > 0: marg_V[x] += r
        #^ for sxyz exists

        # Subtract P(s,x,*,*) * ( P(s,x,*,*) / P(*,x,*,*) )  
        for sx,t in b_sx.items():
            s,x = sx
            if t > 0: mysum -= t*log(t/marg_V[x])
        #^ for sx exists
        itoc = time.process_time()
        if output == 2: print("Time to compute H(S|X):", itoc - itic, "secs")
        return mysum
    #^ if

    # Compute H(S|Y)
    if V == 2:
        itic = time.process_time()
        b_sy = marginal_sy(p)

        # Get P(*,*,y,*)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if r > 0: marg_V[y] += r
        #^ for sxyz exists
        
        # Subtract P(s,*,y,*) * ( P(s,*,y,*) / P(*,*,y,*) )  
        for sy,t in b_sy.items():
            s,y = sy
            if t > 0: mysum -= t*log( t / marg_V[y] )
        #^ for sy exists
        itoc = time.process_time()
        if output ==2: print("Time to compute H(S|Y):", itoc - itic, "secs")
        return mysum
    #^ if

    # Compute H(S|Z)
    if V == 3:
        itic = time.process_time()
        b_sz = marginal_sz(p)

        # Get P(*,*,*,z)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if r > 0: marg_V[z] += r
        #^ for sxyz exists

        # Subtract P(s,*,*,z) * ( P(s,*,*,z) / P(*,*,*,z) )  
        for sz,t in b_sz.items():
            s,z = sz
            if t > 0: mysum -= t*log( t / marg_V[z] )
        #^ for sz exists
        itoc = time.process_time()
        if output == 2: print("Time to compute H(S|Z):", itoc - itic, "secs")
    return mysum
    #^ if
#^ condent_V()

# Compute the Mutual Information MI(W;T)
def I_V(V,p):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    
    # Initialization
    marg_V = defaultdict(lambda: 0.)
    marg_S = defaultdict(lambda: 0.)
    mysum = 0.

    # Compute I(S; X)
    if V == 1:
        b_sx = marginal_sx(p)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            # Get P(s,*,*,*) & P(*,x,*,*)
            if r > 0:
                marg_S[s] += r                
                marg_V[x] += r
        #^ for sxyz exists

        # add P(s,x,*,*)*( P(s,x,*,*) / (P(s,*,*,*) * P(*,x,*,*) )
        for sx,t in b_sx.items():
            s,x = sx
            if t > 0: mysum += t*log( t / ( marg_S[s]*marg_V[x] ) )
        #^ for sx exists
    #^ if

    # Compute I(S; Y)
    if V == 2:
        b_sy = marginal_sy(p)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            # Get P(s,*,*,*) & P(*,*,y,*)
            if r > 0:
                marg_S[s] += r                
                marg_V[y] += r
        #^ for sxyz exists

        # add P(s,*,y,*)*( P(s,*,y,*) / (P(s,*,*,*) * P(*,*,y,*) )
        for sy,t in b_sy.items():
            s,y = sy
            if t > 0: mysum += t*log( t / ( marg_S[s]*marg_V[y] ) )
        #^ for sy exists
    #^ if
    
    # Compute I(S; Z)
    if V == 3:
        b_sz = marginal_sz(p)

        # Get P(s,*,*,*) & P(*,*,*,z)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if r > 0:
                marg_S[s] += r                
                marg_V[z] += r
        #^ for sxyz exists

        # add P(s,*,*,z)*( P(s,*,*,z) / (P(s,*,*,*) * P(*,*,*,z) ) 
        for sz,t in b_sz.items():
            s,z = sz
            if t > 0: mysum += t*log( t / (marg_S[s]*marg_V[z]) )
        #^ for sz exists
    #^ if

    return mysum
#^ I_V()

# def I_X_YZ(p):
#     # Mutual information I( X ; Y , Z )
#     mysum    = 0.
#     marg_x   = defaultdict(lambda: 0.)
#     marg_yz  = defaultdict(lambda: 0.)
#     for xyz,r in p.items():
#         x,y,z = xyz
#         if r > 0 :
#             marg_x[x]      += r
#             marg_yz[(y,z)] += r
    
#     for xyz,t in p.items():
#         x,y,z = xyz
#         if t > 0:  mysum += t * log( t / ( marg_x[x]*marg_yz[(y,z)] ) )
#     return mysum
# #^ I_X_YZ()

def pid(pdf_dirty, cone_solver="ECOS", output=0, parallel="off", **solver_args):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0

    assert type(pdf_dirty) is dict, "chicharro_pid.pid(pdf): pdf must be a dictionary"
    assert type(cone_solver) is str, "chicharro_pid.pid(pdf): `cone_solver' parameter must be string (e.g., 'ECOS')"
    if __debug__:
        sum_p = 0
        for k,v in pdf_dirty.items():
            assert type(k) is tuple or type(k) is list,           "chicharro_pid.pid(pdf): pdf's keys must be tuples or lists"
            assert len(k)==4,                                     "chicharro_pid.pid(pdf): pdf's keys must be tuples/lists of length 4"
            assert type(v) is float or ( type(v)==int and v==0 ), "chicharro_pid.pid(pdf): pdf's values must be floats"
            assert v > -.1,                                       "chicharro_pid.pid(pdf): pdf's values must not be negative"
            sum_p += v
        #^ for
        assert abs(sum_p - 1)< 1.e-10,                                       "chicharro_pid.pid(pdf): pdf's values must sum up to 1 (tolerance of precision is 1.e-10)"
    #^ if
    assert type(output) is int, "chicharro_pid.pid(pdf,output): output must be an integer"

    # Check if the solver is implemented:
    assert cone_solver=="ECOS", "chicharro_pid.pid(pdf): We currently don't have an interface for the Cone Solver "+cone_solver+" (only ECOS)."

    pdf = { k:v  for k,v in pdf_dirty.items() if v > 1.e-300 }

    tic_marg = time.time()
    bx_sx = marginal_sx(pdf)
    by_sy = marginal_sy(pdf)
    bz_sz = marginal_sz(pdf)

    b_xy = marginal_xy(pdf)
    b_xz = marginal_xz(pdf)
    b_yz = marginal_yz(pdf)
    
    toc_marg = time.time()
    if output > 0: print("\nchicharro_pid.pid(): Time to create marginals:", toc_marg - tic_marg, "secs\n")
    # if cone_solver=="ECOS": .....
    if output > 0:  print("\nchicharro_pid.pid(): Preparing Cone Program data",end="...\n")

    solver = Solve_w_ECOS(bx_sx, by_sy, bz_sz, b_xy, b_xz, b_yz)
    subsolver_I = Opt_I(bx_sx, by_sy, bz_sz, b_xy, b_xz, b_yz)
    subsolver_II = Opt_II(bx_sx, by_sy, bz_sz, b_xy, b_xz, b_yz)

    tic_mod = time.time()
        
    if parallel == 'on':
        pool = Pool()
        # call create_model() for min -H( S|XY ), min -H( S|XZ ), min -H( S|YZ )
        cre_12 = pool.apply_async(subsolver_II.create_model, [ [1,2], output ])
        cre_13 = pool.apply_async(subsolver_II.create_model, [ [1,3], output ])
        cre_23 = pool.apply_async(subsolver_II.create_model, [ [2,3], output ])

        # call create_model() for min -H( S|XYZ )
        cre_I = pool.apply_async(subsolver_I.create_model, [ output ])

        # Get the models
        c_I, G_I, h_I, dims_I, A_I, b_I = cre_I.get()
        c_12, G_12, h_12, dims_12, A_12, b_12 = cre_12.get()
        c_13, G_13, h_13, dims_13, A_13, b_13 = cre_13.get()
        c_23, G_23, h_23, dims_23, A_23, b_23 = cre_23.get()
        pool.close()
        pool.join()
    else:
        # create model min -H( S|XYZ )
        c_I, G_I, h_I, dims_I, A_I, b_I = subsolver_I.create_model(output)

        # create models min -H( S|XY ), min -H( S|XZ ), min -H( S|YZ )
        c_12, G_12, h_12, dims_12, A_12, b_12 = subsolver_II.create_model([1,2], output)
        c_13, G_13, h_13, dims_13, A_13, b_13 = subsolver_II.create_model([1,3], output)
        c_23, G_23, h_23, dims_23, A_23, b_23 = subsolver_II.create_model([2,3], output)
    #^ if parallel
    
    toc_mod = time.time()
    if output > 0: print("\nchicharro_pid.pid(): Time to create all models. ", toc_mod - tic_mod, "secs\n")

    if output > 2:
        subsolver_I.verbose = True
        subsolver_II.verbose = True
        

    ecos_keep_solver_obj = False
    if 'keep_solver_object' in solver_args.keys():
        if solver_args['keep_solver_object']==True: ecos_keep_solver_obj = True
        del solver_args['keep_solver_object']

    subsolver_I.ecos_kwargs = solver_args
    subsolver_II.ecos_kwargs = solver_args
    if output > 0: print("chicharro_pid.pid(): Preparing Cone Program data is done.")

    if output == 1: print("chicharro_pid.pid(): Starting solver",end="...")
    if output > 2: print("chicharro_pid.pid(): Starting solver.")

    if parallel == "on":

        # Find the optimal solution of: min H(S|XYZ), min H(S|XY), H(S|XZ) and H(S|YZ)
        pool = Pool()
        # Call solve()  
        res_I = pool.apply_async(subsolver_I.solve, [ c_I, G_I, h_I, dims_I, A_I, b_I, output ]) 
        res_12 = pool.apply_async(subsolver_II.solve,[c_12, G_12, h_12, dims_12, A_12, b_12, output])
        res_13 = pool.apply_async(subsolver_II.solve,[c_13, G_13, h_13, dims_13, A_13, b_13, output])
        res_23 = pool.apply_async(subsolver_II.solve,[c_23, G_23, h_23, dims_23, A_23, b_23, output])

        # Get the optimal solutions 
        retval_I, sol_rpq_I, sol_slack_I, sol_lambda_I, sol_mu_I, sol_info_I = res_I.get()
        retval_12, sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12, sol_info_12 = res_12.get()
        retval_13, sol_rpq_13, sol_slack_13, sol_lambda_13, sol_mu_13, sol_info_13 = res_13.get()
        retval_23, sol_rpq_23, sol_slack_23, sol_lambda_23, sol_mu_23, sol_info_23 = res_23.get()
        pool.close()
        pool.join()
    else:
        # Solve the optimization: min -H( S|XYZ )
        retval_I, sol_rpq_I, sol_slack_I, sol_lambda_I, sol_mu_I, sol_info_I = subsolver_I.solve(c_I, G_I, h_I, dims_I, A_I, b_I, output)

        # Solve the optimizations: min -H( S|XY ), min -H( S|XZ ), min -H( S|YZ )
        retval_12, sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12, sol_info_12 = subsolver_II.solve(c_12, G_12, h_12, dims_12, A_12, b_12, output)
        retval_13, sol_rpq_13, sol_slack_13, sol_lambda_13, sol_mu_13, sol_info_13 = subsolver_II.solve(c_13, G_13, h_13, dims_13, A_13, b_13, output)
        retval_23, sol_rpq_23, sol_slack_23, sol_lambda_23, sol_mu_23, sol_info_23 = subsolver_II.solve(c_23, G_23, h_23, dims_23, A_23, b_23, output)
    #^ if parallel

    # Print warning when  a solution is not returned for any of the optimization
    if retval_I != "success" and retval_12 != "success" and retval_13 != "success" and retval_23 != "success":
        print("\nCone Programming solver failed to find (near) optimal solution.\nPlease report the input probability density function to abdullah.makkeh@gmail.com\n")
        if ecos_keep_solver_obj:
            return solver
        else:
            raise Chicharro_pid_Exception("Chicharro_pid_Exception: Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")
        #^ if (keep solver)
    #^ if (solve failure)

    if output > 0:  print("\nchicharro_pid.pid(): Solving is done.\n")

    tic_stats = time.time()
    if output > 0:
        print("\nchicharro_pid.pid(): Stats for optimizing H(S|X,Y,Z):\n", sol_info_I)
        print("\nchicharro_pid.pid(): Stats for optimizing H(S|X,Y):\n", sol_info_12)
        print("\nchicharro_pid.pid(): Stats for optimizing H(S|X,Z):\n", sol_info_13)
        print("\nchicharro_pid.pid(): Stats for optimizing H(S|Y,Z):\n", sol_info_23)
    if parallel == 'on':

        # Compute the value of the dual objective function for each optimization:
        # min -H( S|XYZ ) min -H( S|XY ), min -H( S|XZ ) and  min -H( S|YZ ) ) 
        pool = Pool()

        # Call dual_value() to compute the 
        dual_I        = pool.apply_async(subsolver_I.dual_value,[sol_lambda_I, b_I])
        dual_12       = pool.apply_async(subsolver_II.dual_value,[sol_lambda_12, b_12])
        dual_13       = pool.apply_async(subsolver_II.dual_value,[sol_lambda_13, b_13])
        dual_23       = pool.apply_async(subsolver_II.dual_value,[sol_lambda_23, b_23])

        # Get the dual value
        dual_val_I    = dual_I.get()
        dual_val_12   = dual_12.get()
        dual_val_13   = dual_13.get()
        dual_val_23   = dual_23.get()
        pool.close()
        pool.join()

        # Compute the marginals of the optimal pdf (part of the optimal solution is a pdf) 
        pool = Pool()

        # Call marginals() 
        marg_12 = pool.apply_async(subsolver_II.marginals,[[1,2], sol_rpq_12, output])
        marg_13 = pool.apply_async(subsolver_II.marginals,[[1,3], sol_rpq_13, output])
        marg_23 = pool.apply_async(subsolver_II.marginals,[[2,3], sol_rpq_23, output])

        # Get the marginals
        marg_12_S, marg_12_X, marg_12_Y, marg_12_Z, marg_12_SX, marg_12_SY, marg_12_SZ, marg_12_XY, marg_12_XZ, marg_12_YZ, marg_12_SXY, marg_12_SXZ, marg_12_SYZ = marg_12.get()
        marg_13_S, marg_13_X, marg_13_Y, marg_13_Z, marg_13_SX, marg_13_SY, marg_13_SZ, marg_13_XY, marg_13_XZ, marg_13_YZ, marg_13_SXY, marg_13_SXZ, marg_13_SYZ = marg_13.get()
        marg_23_S, marg_23_X, marg_23_Y, marg_23_Z, marg_23_SX, marg_23_SY, marg_23_SZ, marg_23_XY, marg_23_XZ, marg_23_YZ, marg_23_SXY, marg_23_SXZ, marg_23_SYZ = marg_23.get()
        pool.close()
        pool.join()


        # Compute H(S|X,Y,Z), H(S|X,Y), H(S|X,Z), H(S|Y,Z) (using the optimal pdf) 
        pool = Pool()

        # Call condentropy() and condentropy_2vars() 
        con_I = pool.apply_async(subsolver_I.condentropy,[sol_rpq_I, output])
        con_12 = pool.apply_async(subsolver_II.condentropy_2vars,[ [1,2], sol_rpq_12, output, marg_12_XY, marg_12_XZ, marg_12_YZ,marg_12_SXY, marg_12_SXZ, marg_12_SYZ])
        con_13        = pool.apply_async(subsolver_II.condentropy_2vars,[[1,3], sol_rpq_13, output, marg_13_XY, marg_13_XZ, marg_13_YZ, marg_13_SXY, marg_13_SXZ, marg_13_SYZ])
        con_23        = pool.apply_async(subsolver_II.condentropy_2vars,[[2,3], sol_rpq_23, output, marg_23_XY, marg_23_XZ, marg_23_YZ, marg_23_SXY, marg_23_SXZ, marg_23_SYZ])

        # Get H(S|X,Y,Z), H(S|X,Y), H(S|X,Z) and H(S|Y,Z)
        condent_I     = con_I.get()
        condent_12    = con_12.get()
        condent_13    = con_13.get()
        condent_23    = con_23.get()
        print( "H(S|XYZ):", condent_I)
        print( "H(S|XY):", condent_12)
        print( "H(S|XZ):", condent_13)
        print( "H(S|YZ):", condent_23)

        pool.close()
        pool.join()

        # Compute H(S), H(S|X), H(S|Y) and H(S|Z) 
        pool = Pool()

        # Call ent_S() and condent_V()
        ent_S = pool.apply_async(solver.entropy_V, [1, pdf, output])
        con_1 = pool.apply_async(condent_V, [1, pdf, output])
        con_2 = pool.apply_async(condent_V, [2, pdf, output])
        con_3 = pool.apply_async(condent_V, [3, pdf, output])

        # Get H(S), H(S|X), H(S|Y) and H(S|Z) 
        entropy_S     = ent_S.get()

        condent_1     = con_1.get()
        condent_2     = con_2.get()
        condent_3     = con_3.get()
        pool.close()
        pool.join()

        # Compute H(S|X,Y,Z) of the original pdf (not optimal pdf)
        condent__orig = solver.condentropy__orig(pdf, output)
    else:

        # Compute the value of the dual objective function for each optimization:
        # min -H( S|XYZ ) min -H( S|XY ), min -H( S|XZ ) and  min -H( S|YZ )
        dual_val_I    = subsolver_I.dual_value(sol_lambda_I, b_I)
        dual_val_12   = subsolver_II.dual_value(sol_lambda_12, b_12)
        dual_val_13   = subsolver_II.dual_value(sol_lambda_13, b_13)
        dual_val_23   = subsolver_II.dual_value(sol_lambda_23, b_23)

        # Compute the marginals of the optimal pdf (part of the optimal solution is a pdf) 
        marg_12_S, marg_12_X, marg_12_Y, marg_12_Z, marg_12_SX, marg_12_SY, marg_12_SZ, marg_12_XY, marg_12_XZ, marg_12_YZ, marg_12_SXY, marg_12_SXZ, marg_12_SYZ = subsolver_II.marginals([1,2], sol_rpq_12, output)
        
        marg_13_S, marg_13_X, marg_13_Y, marg_13_Z, marg_13_SX, marg_13_SY, marg_13_SZ, marg_13_XY, marg_13_XZ, marg_13_YZ, marg_13_SXY, marg_13_SXZ, marg_13_SYZ = subsolver_II.marginals([1,3], sol_rpq_13, output)

        marg_23_S, marg_23_X, marg_23_Y, marg_23_Z, marg_23_SX, marg_23_SY, marg_23_SZ, marg_23_XY, marg_23_XZ, marg_23_YZ, marg_23_SXY, marg_23_SXZ, marg_23_SYZ = subsolver_II.marginals([2,3], sol_rpq_23, output)


        # Compute H(S|X,Y,Z), H(S|X,Y), H(S|X,Z), H(S|Y,Z) (using the optimal pdf) 
        condent_I     = subsolver_I.condentropy(sol_rpq_I, output)
        condent_12    = subsolver_II.condentropy_2vars([1,2], sol_rpq_12, output, marg_12_XY, marg_12_XZ, marg_12_YZ,marg_12_SXY, marg_12_SXZ, marg_12_SYZ)
        condent_13    = subsolver_II.condentropy_2vars([1,3],sol_rpq_13, output, marg_13_XY, marg_13_XZ, marg_13_YZ,marg_13_SXY, marg_13_SXZ, marg_13_SYZ)
        condent_23    = subsolver_II.condentropy_2vars([2,3], sol_rpq_23, output, marg_23_XY, marg_23_XZ, marg_23_YZ,marg_23_SXY, marg_23_SXZ, marg_23_SYZ)

        # Compute H(S), H(S|X), H(S|Y) and H(S|Z) 
        entropy_S     = solver.entropy_V(1, pdf, output)
        condent_1     = condent_V(1, pdf, output)
        condent_2     = condent_V(2, pdf, output)
        condent_3     = condent_V(3, pdf, output)

        # Compute H(S|X,Y,Z) of the original pdf (not optimal pdf) 
        condent__orig = solver.condentropy__orig(pdf, output)

    # elsif cone_solver=="SCS":
    # .....
    # #^endif
    toc_stats = time.time()
    if output > 0: print("\nchicharro_pid.pid(): Time for retrieving results:", toc_stats - tic_stats, "secs\n")


    tic_dict = time.time()

    # Compute the PID quantities  
    return_data = dict()
    return_data["CI"]    = condent_I  - condent__orig
    return_data["UIX"]   = condent_23 - condent_I
    return_data["UIY"]   = condent_13 - condent_I
    return_data["UIZ"]   = condent_12 - condent_I
    return_data["UIXY"]  = condent_I  + condent_3     - condent_13        - condent_23
    return_data["UIXZ"]  = condent_I  + condent_2     - condent_12        - condent_23
    return_data["UIYZ"]  = condent_I  + condent_1     - condent_12        - condent_13
    return_data["SI"]    = entropy_S  - condent__orig - return_data["CI"] - return_data["UIX"]  - return_data["UIY"]  - return_data["UIZ"] - return_data["UIXY"] - return_data["UIXZ"] - return_data["UIYZ"]

    tic_o = time.time()    
    if parallel == "on":

        # Compute the feasibility violations of the optimization problems:
        # min -H( S|XYZ ) min -H( S|XY ), min -H( S|XZ ) and  min -H( S|YZ ) ) 
        pool = Pool()

        # Call check_Feasibility()
        feas_I = pool.apply_async(subsolver_I.check_feasibility, [ sol_rpq_I,sol_lambda_I,output ])

        feas_12 = pool.apply_async(subsolver_II.check_feasibility, [ [1,2], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12,output ])

        feas_13 = pool.apply_async(subsolver_II.check_feasibility, [ [1,3], sol_rpq_13, sol_slack_13, sol_lambda_13, sol_mu_13,output ])

        feas_23 = pool.apply_async(subsolver_II.check_feasibility, [ [2,3], sol_rpq_23, sol_slack_23, sol_lambda_23, sol_mu_23,output ])

        # Get feasibility violations
        primal_infeas_I,dual_infeas_I = feas_I.get()
        primal_infeas_12,dual_infeas_12 = feas_12.get()
        primal_infeas_13,dual_infeas_13 = feas_13.get()
        primal_infeas_23,dual_infeas_23 = feas_23.get()
        pool.close()
        pool.join()
    else:
        # Compute the feasibility violations of the optimization problems:
        # min -H( S|XYZ ) min -H( S|XY ), min -H( S|XZ ) and  min -H( S|YZ ) ) 

        primal_infeas_I,dual_infeas_I = subsolver_I.check_feasibility(sol_rpq_I,sol_lambda_I,output)
        

        primal_infeas_12,dual_infeas_12 = subsolver_II.check_feasibility([1,2], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12,output)
        

        primal_infeas_13,dual_infeas_13 = subsolver_II.check_feasibility([1,3], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12,output)
        

        primal_infeas_23,dual_infeas_23 = subsolver_II.check_feasibility([2,3], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12,output)
    #^if parallel

    toc_o = time.time()
    if output > 0: print("\nchicharro_pid.pid(): Time for computing Numerical Errors:", toc_o - tic_o, "secs\n")

    # Store the numerical violations of the optimization problems:
    # min -H( S|XYZ ) min -H( S|XY ), min -H( S|XZ ) and  min -H( S|YZ ) )
    
    return_data["Num_err_I"] = (primal_infeas_I, dual_infeas_I, max(-condent_I*ln(2) - dual_val_I, 0.0))
    return_data["Num_err_12"] = (primal_infeas_12,dual_infeas_12, max(-condent_12*ln(2) - dual_val_12, 0.0))
    return_data["Num_err_13"] = (primal_infeas_13,dual_infeas_13, max(-condent_13*ln(2) - dual_val_13, 0.0))
    return_data["Num_err_23"] = (primal_infeas_23,dual_infeas_23, max(-condent_23*ln(2) - dual_val_23, 0.0))
    return_data["Solver"] = "ECOS http://www.embotech.com/ECOS"

    # Store the model and the problem if asked to 
    if ecos_keep_solver_obj:
        return_data["Ecos Solver Object"] = solver
        return_data["Opt I Solver Object"] = subsolver_I
        return_data["Opt II Solver Object"] = subsolver_II
        
    #^ if (keep solver)
    
    toc_dict = time.time()
    if output > 0: print("\nchicharro_pid.pid(): Time for storing results:", toc_dict - tic_dict, "secs\n")

    # Sanity check

    # Check: MI(S; X,Y,Z) = SI + CI + UIX + UIY + UIZ + UIXY + UIXZ + UIYZ
    assert abs(entropy_S - condent__orig
               - return_data['CI'] - return_data['SI']
               - return_data['UIX'] - return_data['UIY'] - return_data['UIZ']
               - return_data['UIXY'] - return_data['UIXZ'] - return_data['UIYZ'])< 1.e-10,                                 "chicharro_pid.pid(): PID quantities must  sum up to mutual information (tolerance of precision is 1.e-10)"

    # Check: MI(S; X) = SI + UIX + UIXY + UIXZ
    assert abs( I_V(1,pdf)
                - return_data['SI'] - return_data['UIX'] - return_data['UIXY'] - return_data['UIXZ'] ) < 1.e-10,           "chicharro_pid.pid(): Unique and shared of X must sum up to MI(S; X) (tolerance of precision is 1.e-10)"

    # Check: MI(S; Y) = SI + UIY + UIXY + UIYZ
    assert abs( I_V(2,pdf)
                - return_data['SI'] - return_data['UIY'] - return_data['UIXY'] - return_data['UIYZ'] ) < 1.e-10,           "chicharro_pid.pid(): Unique and shared of Y must sum up to MI(S; Y) (tolerance of precision is 1.e-10)"


    # Check: MI(S; Z) = SI + UIZ + UIXZ + UIYZ
    assert abs( I_V(3,pdf)
                - return_data['SI'] - return_data['UIZ'] - return_data['UIXZ'] - return_data['UIYZ'])< 1.e-10,             "chicharro_pid.pid(): Unique and shared of Z must sum up to MI(S; Z) (tolerance of precision is 1.e-10)"

    return return_data
#^ pid()

#EOF
