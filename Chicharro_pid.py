# Chicharro_pid.py -- Python module
#
# Chicharro_pid: Chicharro trivariate Partial Information Decomposition
# https://github.com/Abzinger/Chicharro_pid 
# (c) Abdullah Makkeh, Dirk Oliver Theis
# Permission to use and modify with proper attribution
# (Apache License version 2.0)
#
# Information about the algorithm, documentation, and examples are here:
# @Article{makkeh2018broja,
#          author =       {Makkeh, Abdullah and Theis, Dirk Oliver and Vicente, Raul},
#          title =        {BROJA-2PID: A cone programming based Partial Information Decomposition estimator},
#          journal =      {Entropy},
#          year =         2018,
#          volume =    {20},
#          number =    {4},
#          pages =     {271}
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

log = math.log2
ln  = math.log
# ECOS's exp cone: (r,p,q)   w/   q>0  &  exp(r/q) ≤ p/q
# Translation:     (0,1,2)   w/   2>0  &  0/2      ≤ ln(1/2)
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
    def __init__(self, marg_sx, marg_sy, marg_sz):
        # (c) Abdullah Makkeh, Dirk Oliver Theis
        # Permission to use and modify under Apache License version 2.0

        # ECOS parameters
        self.ecos_kwargs   = dict()
        self.verbose       = False

        # # Data for ECOS
        # self.c            = None
        # self.G            = None
        # self.h            = None
        # self.dims         = dict()
        # self.A            = None
        # self.b            = None

        # # ECOS result
        # self.sol_rpq    = None
        # self.sol_slack  = None #
        # self.sol_lambda = None # dual variables for equality constraints
        # self.sol_mu     = None # dual variables for generalized ieqs
        # self.sol_info   = None

        # Probability density funciton data
        self.b_sx         = dict(marg_sx)
        self.b_sy         = dict(marg_sy)
        self.b_sz         = dict(marg_sz)
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

    def condentropy__orig(self,pdf):
        mysum = 0.
        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    s_list = [ s  for s in self.S if (s,x,y,z) in pdf.keys()]
                    marg = 0.
                    for s in s_list: marg += pdf[(s,x,y,z)]
                    for s in s_list:
                        p = pdf[(s,x,y,z)]
                        mysum -= p*log(p/marg)
                    #^ for xyz
                #^ for z
            #^ for y
        #^ for x
        return mysum
    #^ condentropy__orig()

#^ class Solve_w_ECOS

class Opt_I(Solve_w_ECOS):
    # def __init__(self): TRIVARIATE_SYN.init(self)

    def create_model(self): TRIVARIATE_SYN.create_model(self)
    
    def solve(self):
        return TRIVARIATE_SYN.solve(self)
    
    def dual_value(self, sol_lambda):
        return TRIVARIATE_SYN.dual_value(self, sol_lambda)

    def check_feasibility(self, sol_rpq, sol_lambda):
        return TRIVARIATE_SYN.check_feasibility(self, sol_rpq, sol_lambda)
    
    def condentropy(self, sol_rpq):
        return TRIVARIATE_SYN.condentropy(self, sol_rpq)
#^ subclass Opt_I

class Opt_II(Solve_w_ECOS):
    # def __init__(self): TRIVARIATE_SYN.init(self)

    def initialization(self, which_sources):
        return TRIVARIATE_UNQ.initialization(self, which_sources)

    def sq_vidx(self, i, which_sources):
        return TRIVARIATE_UNQ.sq_vidx(self, i, which_sources)

    def marginal_a(self,_A, _B, _C, _D, which_sources, sol_rpq):
        return TRIVARIATE_UNQ.marginal_a(self, _A, _B, _C, _D, which_sources,sol_rpq)

    def marginal_ab(self,_A, _B, _C, _D, which_sources,sol_rpq):
        return TRIVARIATE_UNQ.marginal_ab(self, _A, _B, _C, _D, which_sources,sol_rpq)

    def marginal_abc(self,_A, _B, _C, _D, which_sources,sol_rpq):
        return TRIVARIATE_UNQ.marginal_abc(self,_A, _B, _C, _D, which_sources,sol_rpq)

    def create_model(self, which_sources):
        return TRIVARIATE_UNQ.create_model(self, which_sources)
    
    def solve(self, c, G, h, dims, A, b):
        return TRIVARIATE_UNQ.solve(self, c, G, h, dims, A, b)
    
    def dual_value(self, sol_lambda, b):
        return TRIVARIATE_UNQ.dual_value(self, sol_lambda, b)

    def check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu):
        return TRIVARIATE_UNQ.check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu)
    
    def condentropy_2vars(self, which_sources,sol_rpq):
        return TRIVARIATE_UNQ.condentropy_2vars(self, which_sources,sol_rpq)

    def condentropy_1var(self, which_sources, sol_rpq):
        return TRIVARIATE_UNQ.condentropy_1var(self, which_sources, sol_rpq)

    def entropy_S(self, pdf):
        return TRIVARIATE_UNQ.entropy_S(self, pdf)
#^ subclass Opt_II

def marginal_sx(p):
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (s,x) in marg.keys():    marg[(s,x)] += r
        else:                       marg[(s,x)] =  r
    return marg

def marginal_sy(p):
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (s,y) in marg.keys():   marg[(s,y)] += r
        else:                      marg[(s,y)] =  r
    return marg

def marginal_sz(p):
    marg = dict()
    for sxyz,r in p.items():
        s,x,y,z = sxyz
        if (s,z) in marg.keys():   marg[(s,z)] += r
        else:                      marg[(s,z)] =  r
    return marg

def condent_V(V, p):
    marg_V = dict()
    mysum = 0.
    if V == 1:
        b_sx = marginal_sx(p)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if x in marg_V.keys(): marg_V[x] += r
            else:                  marg_V[x]  = r
        #^ for sxyz exists
        for sx,t in b_sx.items():
            s,x = sx
            if t > 0: mysum -= t*log(t/marg_V[x])
        #^ for sx exists
        return mysum
    #^ if
    if V == 2:
        b_sy = marginal_sy(p)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if y in marg_V.keys(): marg_V[y] += r
            else:                  marg_V[y]  = r
        #^ for sxyz exists
        for sy,t in b_sy.items():
            s,y = sy
            if t > 0: mysum -= t*log(t/marg_V[y])
        #^ for sx exists
        return mysum
    #^ if
    if V == 3:
        b_sz = marginal_sz(p)
        for sxyz,r in p.items():
            s,x,y,z = sxyz
            if z in marg_V.keys(): marg_V[z] += r
            else:                  marg_V[z]  = r
        #^ for sxyz exists
        for sz,t in b_sz.items():
            s,z = sz
            if t > 0: mysum -= t*log(t/marg_V[z])
        #^ for sx exists
    return mysum
    #^ if
# def entropy_S(S, p):
#     mysum = 0.
#     marg_S = dict()
#     for sxyz,r in p.items():
#         s,x,y,z = sxyz
#         if s in marg_S.keys(): marg_S[s] += r
#         else:                  marg_S[s]  = r
#     #^ for sxyz exists
#     for s in S:
#         mysum -= marg_S[s]*log(marg_S[s])
#     #^ for x
#     return mysum
# #^ entropy_S()

    
    # def I_X_Y(p):
#     # Mutual information I( X ; Y )
#     mysum   = 0.
#     marg_x  = defaultdict(lambda: 0.)
#     marg_y  = defaultdict(lambda: 0.)
#     b_xy    = marginal_xy(p)
#     for xyz,r in p.items():
#         x,y,z = xyz
#         if r > 0 :
#             marg_x[x] += r
#             marg_y[y] += r
    
#     for xy,t in b_xy.items():
#         x,y = xy
#         if t > 0:  mysum += t * log( t / ( marg_x[x]*marg_y[y] ) )
#     return mysum
# #^ I_X_Y()

# def I_X_Z(p):
#     # Mutual information I( X ; Z )
#     mysum   = 0.
#     marg_x  = defaultdict(lambda: 0.)
#     marg_z  = defaultdict(lambda: 0.)
#     b_xz    = marginal_xz(p)
#     for xyz,r in p.items():
#         x,y,z = xyz
#         if r > 0 :
#             marg_x[x] += r
#             marg_z[z] += r
    
#     for xz,t in b_xz.items():
#         x,z = xz
#         if t > 0:  mysum += t * log( t / ( marg_x[x]*marg_z[z] ) )
#     return mysum
# #^ I_X_Z()

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
        for k,v in pdf_dirty.items():
            assert type(k) is tuple or type(k) is list,           "chicharro_2pid.pid(pdf): pdf's keys must be tuples or lists"
            assert len(k)==4,                                     "chicharro_pid.pid(pdf): pdf's keys must be tuples/lists of length 4"
            assert type(v) is float or ( type(v)==int and v==0 ), "chicharro_pid.pid(pdf): pdf's values must be floats"
            assert v > -.1,                                       "chicharro_pid.pid(pdf): pdf's values must not be negative"
        #^ for
    #^ if
    assert type(output) is int, "chicharro_pid.pid(pdf,output): output must be an integer"

    # Check if the solver is implemented:
    assert cone_solver=="ECOS", "chicharro_pid.pid(pdf): We currently don't have an interface for the Cone Solver "+cone_solver+" (only ECOS)."

    pdf = { k:v  for k,v in pdf_dirty.items() if v > 1.e-300 }

    bx_sx = marginal_sx(pdf)
    by_sy = marginal_sy(pdf)
    bz_sz = marginal_sz(pdf)

    # if cone_solver=="ECOS": .....
    if output > 0:  print("BROJA_2PID: Preparing Cone Program data",end="...")
    solver = Solve_w_ECOS(bx_sx, by_sy, bz_sz)
    subsolver_I = Opt_I(bx_sx, by_sy, bz_sz)
    subsolver_II = Opt_II(bx_sx, by_sy, bz_sz)
    subsolver_I.create_model()
    c_12, G_12, h_12, dims_12, A_12, b_12 = subsolver_II.create_model([1,2])
    c_13, G_13, h_13, dims_13, A_13, b_13 = subsolver_II.create_model([1,3])
    c_23, G_23, h_23, dims_23, A_23, b_23 = subsolver_II.create_model([2,3])
    
    if output > 1:
        subsolver_I.verbose = True
        subsolver_II.verbose = True
        

    ecos_keep_solver_obj = False
    if 'keep_solver_object' in solver_args.keys():
        if solver_args['keep_solver_object']==True: ecos_keep_solver_obj = True
        del solver_args['keep_solver_object']

    subsolver_I.ecos_kwargs = solver_args
    subsolver_II.ecos_kwargs = solver_args
    if output > 0: print("done.")

    if output == 1: print("Chicharro_pid: Starting solver",end="...")
    if output > 1: print("Chicharro_pid: Starting solver.")

    if parallel == "on":
        pool = Pool(8)
        # res_t  = pool.apply_async(Solve_w_ECOS,[bx_sx, by_sy, bz_sz])
        # solver = res_t.get()
        # res_tt = pool.apply_async(Opt_I,[bx_sx, by_sy, bz_sz])
        # subsolver_I = res_tt.get()
        # cre_t  =pool.apply_async(subsolver_I.create_model)
        # cre_t.get()
        res_I = pool.apply_async(subsolver_I.solve)
        retval_I, sol_rpq_I, sol_slack_I, sol_lambda_I, sol_mu_I, sol_info_I = res_I.get()
        res_12 = pool.apply_async(subsolver_II.solve,[c_12, G_12, h_12, dims_12, A_12, b_12])
        retval_12, sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12, sol_info_12 = res_12.get()
        res_13 = pool.apply_async(subsolver_II.solve,[c_13, G_13, h_13, dims_13, A_13, b_13])
        retval_13, sol_rpq_13, sol_slack_13, sol_lambda_13, sol_mu_13, sol_info_13 = res_13.get()
        res_23 = pool.apply_async(subsolver_II.solve,[c_23, G_23, h_23, dims_23, A_23, b_23])
        retval_23, sol_rpq_23, sol_slack_23, sol_lambda_23, sol_mu_23, sol_info_23 = res_23.get()
        pool.close()
    else:
        retval_I, sol_rpq_I, sol_slack_I, sol_lambda_I, sol_mu_I, sol_info_I = subsolver_I.solve()
        retval_12, sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12, sol_info_12 = subsolver_II.solve(c_12, G_12, h_12, dims_12, A_12, b_12)
        retval_13, sol_rpq_13, sol_slack_13, sol_lambda_13, sol_mu_13, sol_info_13 = subsolver_II.solve(c_13, G_13, h_13, dims_13, A_13, b_13)
        retval_23, sol_rpq_23, sol_slack_23, sol_lambda_23, sol_mu_23, sol_info_23 = subsolver_II.solve(c_23, G_23, h_23, dims_23, A_23, b_23)
    #^ if parallel
    
    if retval_I != "success" and retval_12 != "success" and retval_13 != "success" and retval_23 != "success":
        print("\nCone Programming solver failed to find (near) optimal solution.\nPlease report the input probability density function to abdullah.makkeh@gmail.com\n")
        if ecos_keep_solver_obj:
            return solver
        else:
            raise Chicharro_pid_Exception("Chicharro_pid_Exception: Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")
        #^ if (keep solver)
    #^ if (solve failure)

    if output > 0:  print("\nChicharro_pid: done.")

    if output > 1:
        print("Stats for optimizing H(S|X,Y,Z):\n", sol_info_I)
        print("Stats for optimizing H(S|X,Y):\n", sol_info_12)
        print("Stats for optimizing H(S|X,Z):\n", sol_info_13)
        print("Stats for optimizing H(S|Y,Z):\n", sol_info_23)

    condent_I     = subsolver_I.condentropy(sol_rpq_I)
    dual_val_I    = subsolver_I.dual_value(sol_lambda_I)
    
    condent_12    = subsolver_II.condentropy_2vars([1,2], sol_rpq_12)
    dual_val_12   = subsolver_II.dual_value(sol_lambda_12, b_12)
    
    condent_13    = subsolver_II.condentropy_2vars([1,3],sol_rpq_13)
    dual_val_13   = subsolver_II.dual_value(sol_lambda_13, b_13)
    
    condent_23    = subsolver_II.condentropy_2vars([2,3],sol_rpq_23)
    dual_val_23   = subsolver_II.dual_value(sol_lambda_23,b_23)

    entropy_S     = subsolver_II.entropy_S(pdf)
    condent_1     = condent_V(1,pdf)
    condent_2     = condent_V(2,pdf)
    condent_3     = condent_V(3,pdf)
    condent__orig = solver.condentropy__orig(pdf)

    # elsif cone_solver=="SCS":
    # .....
    # #^endif


    return_data = dict()
    return_data["CI"]    = condent_I  - condent__orig
    return_data["UIX"]   = condent_23 - condent_I
    return_data["UIY"]   = condent_13 - condent_I
    return_data["UIZ"]   = condent_12 - condent_I
    return_data["UIXY"]  = condent_I  + condent_3     - condent_13        - condent_23
    return_data["UIXZ"]  = condent_I  + condent_2     - condent_12        - condent_23
    return_data["UIYZ"]  = condent_I  + condent_1     - condent_12        - condent_13
    return_data["SI"]    = entropy_S  - condent__orig - return_data["CI"] - return_data["UIX"]  - return_data["UIX"]  - return_data["UIX"] - return_data["UIXY"] - return_data["UIXZ"] - return_data["UIYZ"]


    primal_infeas_I,dual_infeas_I = subsolver_I.check_feasibility(sol_rpq_I,sol_lambda_I)
    return_data["Num_err_I"] = (primal_infeas_I, dual_infeas_I, max(-condent_I*ln(2) - dual_val_I, 0.0))

    primal_infeas_12,dual_infeas_12 = subsolver_II.check_feasibility([1,2], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12)
    return_data["Num_err_12"] = (primal_infeas_12,dual_infeas_12, max(-condent_12*ln(2) - dual_val_12, 0.0))

    primal_infeas_13,dual_infeas_13 = subsolver_II.check_feasibility([1,3], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12)
    return_data["Num_err_13"] = (primal_infeas_13,dual_infeas_13, max(-condent_13*ln(2) - dual_val_13, 0.0))

    primal_infeas_23,dual_infeas_23 = subsolver_II.check_feasibility([2,3], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12)
    return_data["Num_err_23"] = (primal_infeas_23,dual_infeas_23, max(-condent_23*ln(2) - dual_val_23, 0.0))

    return_data["Solver"] = "ECOS http://www.embotech.com/ECOS"

    if ecos_keep_solver_obj:
        return_data["Solver Object"] = solver
    #^ if (keep solver)

    return return_data
#^ pid()

#EOF
