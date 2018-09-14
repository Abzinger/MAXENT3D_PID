# TRIVARIATE_UNQ.py
import ecos
from scipy import sparse
import numpy as np
from numpy import linalg as LA
from collections import defaultdict
import math
import time
# from collections import defaultdict
ln  = math.log
log = math.log2
# Creates the optimization problem needed to compute both synergy (also needed for uniqueness)

def core_initialization(_S, X_1, X_2, b_sx1, b_sx2, idx_of, of_idx):
    for s in _S:
        for x1 in X_1:
            if (s,x1) in b_sx1.keys():
                for x2 in X_2:
                    if (s,x2) in b_sx2.keys():
                        idx_of[ (s,x1,x2) ] = len(of_idx)
                        of_idx.append( (s,x1,x2) )
                    #^ if sx2
                #^ for x2
            #^if sx1
        #^ for x1
    #^ for s
def initialization(self, which_sources):
    idx_of_trip  = dict()
    trip_of_idx  = []

    if which_sources == [1,2]:
        core_initialization(self.S, self.X, self.Y, self.b_sx, self.b_sy, idx_of_trip, trip_of_idx)
    elif which_sources == [1,3]:
        core_initialization(self.S, self.X, self.Z, self.b_sx, self.b_sz, idx_of_trip, trip_of_idx)
    elif which_sources == [2,3]:
        core_initialization(self.S, self.Y, self.Z, self.b_sy, self.b_sz, idx_of_trip, trip_of_idx)
    return idx_of_trip, trip_of_idx
#^ initialization 

# ECOS's exp cone: (r,p,w)     w/   w>0  &  exp(r/w) ≤ p/w
# Variables here:  (r,p,w)U(q)
# Translation:     (0,1,2,3)   w/   2>0  &  0/2      ≤ ln(1/2)
def sr_vidx(i):
    return 3*i
def sp_vidx(i):
    return 3*i+1
def sw_vidx(i):
    return 3*i+2
def sq_vidx(self, i, ltrip_of_idx):
    return 3*ltrip_of_idx + i

def create_model(self, which_sources):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0

    # Initialize which sources for the model
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    m = len(self.b_sx) + len(self.b_sy) + len(self.b_sz)
    n = len(trip_of_idx)
    ltrip_of_idx = n
    n_vars = 3*n + len(self.quad_of_idx)
    n_cons = 2*n + m
    
    # Create the equations: Ax = b
    self.b = np.zeros((n_cons,),dtype=np.double)
    
    Eqn   = []
    Var   = []
    Coeff = []

    # The q-w coupling eqautions:
    #         if Sources = X,Y  q_{stv*} - w_{stv} = 0
    #         if Sources = X,Z  q_{st*v} - w_{stv} = 0
    #         if Sources = Y,Z  q_{s*tv} - w_{stv} = 0
    tic_w = time.process_time()
    for i,stv in enumerate(trip_of_idx):
        eqn   = i
        w_var = sw_vidx(i)
        Eqn.append( eqn )
        Var.append( w_var )
        Coeff.append( -1. )
        (s,t,v) = stv
        if which_sources == [1,2]:
            for u in self.Z:
                if (s,t,v,u) in self.idx_of_quad.keys(): 
                    q_var = self.sq_vidx(self.idx_of_quad[ (s,t,v,u) ], ltrip_of_idx)
                    Eqn.append( eqn )
                    Var.append( q_var )
                    Coeff.append( +1. )
                #^ if q_{stv*}
            #^ loop *xy*                
        #^ if SXY
        elif which_sources == [1,3]:  
            for u in self.Y:
                if (s,t,u,v) in self.idx_of_quad.keys(): 
                    q_var = self.sq_vidx(self.idx_of_quad[ (s,t,u,v) ], ltrip_of_idx)
                    Eqn.append( eqn )
                    Var.append( q_var )
                    Coeff.append( +1. )
                #^ if q_{st*v}
            #^ loop *x*z                
        #^ if SXZ
        elif which_sources == [2,3]:
            for u in self.X:
                if (s,u,t,v) in self.idx_of_quad.keys(): 
                    q_var = self.sq_vidx(self.idx_of_quad[ (s,u,t,v) ], ltrip_of_idx)
                    Eqn.append( eqn )
                    Var.append( q_var )
                    Coeff.append( +1. )
                #^ if q_{s*tv}
            #^ loop **yz                
        #^if SYZ
    #^ for stv
    toc_w = time.process_time()
    # print("Done w_{stv}. ", toc_w - tic_w, "secs")
    # running number
    eqn = -1 + len(trip_of_idx)
    
    # The q-p coupling equations:
    #         if Sources = X,Y  q_{*tv*} - p_{stv} = 0
    #         if Sources = X,Z  q_{*t*v} - p_{stv} = 0
    #         if Sources = Y,Z  q_{**tv} - p_{stv} = 0
    # ( Expensive step )
    tic_p = time.process_time()
    for i,stv in enumerate(trip_of_idx):
        eqn     += 1
        p_var   = sp_vidx(i)
        Eqn.append( eqn )
        Var.append( p_var )
        Coeff.append( -1. )
        
        (s,t,v) = stv
        if which_sources == [1,2]:
            if (t,v) in self.b_xy.keys():
                for u1 in self.S:
                    for u2 in self.Z:
                        if (u1,t,v,u2) in self.idx_of_quad.keys(): 
                            q_var = self.sq_vidx(self.idx_of_quad[ (u1,t,v,u2) ], ltrip_of_idx)
                            Eqn.append( eqn )
                            Var.append( q_var )
                            Coeff.append( +1. )
                        #^ if q_{*tv*}
                    #^ for u2
                #^ for u2
            #^ if *xy* exists
        #^ if SXY
        elif which_sources == [1,3]:
            if (t,v) in self.b_xz.keys():
                for u1 in self.S:
                    for u2 in self.Y: 
                        if (u1,t,u2,v) in self.idx_of_quad.keys(): 
                            q_var = self.sq_vidx(self.idx_of_quad[ (u1,t,u2,v) ], ltrip_of_idx)
                            Eqn.append( eqn )
                            Var.append( q_var )
                            Coeff.append( +1. )
                        #^ if q_{*t*v}
                    #^ for u2
                #^ for u1
            #^ if *x*z exists
        #^ if SXZ
        elif which_sources == [2,3]:
            if (t,v) in self.b_yz.keys():
                for u1 in self.S:
                    for u2 in self.X: 
                        if (u1,u2,t,v) in self.idx_of_quad.keys(): 
                            q_var = self.sq_vidx(self.idx_of_quad[ (u1,u2,t,v) ], ltrip_of_idx)
                            Eqn.append( eqn )
                            Var.append( q_var )
                            Coeff.append( +1. )
                        #^ if q_{**tv}
                    #^ for u2
                #^ for u1
            #^ if **yz exists
        #^ if SYZ
    #^ for stv
    toc_p = time.process_time()
    # print("Done t_{stv}. ", toc_p - tic_p, "secs")
    tic_m = time.process_time()
    # The sx marginals q_{sx**} = b^x_{sx}
    for s in self.S:
        for x in self.X:
            if (s,x) in self.b_sx.keys():
                eqn += 1
                for y in self.Y:
                    for z in self.Z:
                        if (s,x,y,z) in self.idx_of_quad.keys():
                            q_var = self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], ltrip_of_idx)
                            Eqn.append( eqn )
                            Var.append( q_var )
                            Coeff.append( 1. )
                        #^ if sxyz exists
                        self.b[eqn] = self.b_sx[ (s,x) ]
                    #^ for z
                #^ for y
            #^ if sx exists
        #^ for x
    #^ for s

    # The sy marginals q_{s*y*} = b^y_{sy}
    for s in self.S:
        for y in self.Y:
            if (s,y) in self.b_sy.keys():
                eqn += 1
                for x in self.X:
                    for z in self.Z:
                        if (s,x,y,z) in self.idx_of_quad.keys():
                            q_var = self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], ltrip_of_idx)
                            Eqn.append( eqn )
                            Var.append( q_var )
                            Coeff.append( 1. )
                        #^ if sxyz exists
                        self.b[eqn] = self.b_sy[ (s,y) ]
                    #^ for z
                #^ for x
            #^ if sy exists
        #^ for y
    #^ for s

    # The sz marginals q_{s**z} = b^z_{sz}
    for s in self.S:
        for z in self.Z:
            if (s,z) in self.b_sz.keys():
                eqn += 1
                for x in self.X:
                    for y in self.Y:
                        if (s,x,y,z) in self.idx_of_quad.keys():
                            q_var = self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], ltrip_of_idx)
                            Eqn.append( eqn )
                            Var.append( q_var )
                            Coeff.append( 1. )
                        #^ if sxyz exits
                        self.b[eqn] = self.b_sz[ (s,z) ]
                    #^ for y
                #^ for x
            #^ if sz exists
        #^ for z
    #^ for s
    toc_m = time.process_time()
    # print("Done marginals. ", toc_m - tic_m, "secs")
    self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(n_cons,n_vars), dtype=np.double)
    
    # Generalized ieqs: gen.nneg of the variable triple (r_i,w_i,p_i), i=0,dots,n-1: 
    Ieq   = []
    Var   = []
    Coeff = []

    # Adding q_{s,x,y,z} >= 0 or q_{s,x,y,z} is free variable

    for i,sxyz in enumerate(self.quad_of_idx):
        q_var = self.sq_vidx(i, ltrip_of_idx)
        Ieq.append( len(Ieq) )
        Var.append( q_var )
        Coeff.append( -1. )
    #^ for sxyz

    for i,stv in enumerate(trip_of_idx):
        r_var = sr_vidx(i)
        w_var = sw_vidx(i)
        p_var = sp_vidx(i)
        
        Ieq.append( len(Ieq) )
        Var.append( r_var )
        Coeff.append( -1. )
        
        Ieq.append( len(Ieq) )
        Var.append( p_var )
        Coeff.append( -1. )
        
        Ieq.append( len(Ieq) )
        Var.append( w_var )
        Coeff.append( -1. )
    #^ for stv

    self.G         = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(n_vars,n_vars), dtype=np.double)
    self.h         = np.zeros( (n_vars,),dtype=np.double )
    self.dims = dict()
    self.dims['e'] = n
    self.dims['l'] = len(self.quad_of_idx)
    
    # Objective function:
    self.c = np.zeros( (n_vars,),dtype=np.double )
    for i,stv in enumerate(trip_of_idx):
        self.c[ sr_vidx(i) ] = -1.
    #^ for stv

    return self.c, self.G, self.h, self.dims, self.A, self.b

#^ create_model()

def solve(self, c, G, h, dims, A, b):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0
    self.marg_xyz = None # for cond[]mutinf computation below
    
    if self.verbose != None:
        # print(self.verbose)
        self.ecos_kwargs["verbose"] = self.verbose
    #^ if
    
    solution = ecos.solve(c, G, h, dims, A, b, **self.ecos_kwargs)

    if 'x' in solution.keys():
        self.sol_rpq    = solution['x']
        self.sol_slack  = solution['s']
        self.sol_lambda = solution['y']
        self.sol_mu     = solution['z']
        self.sol_info   = solution['info']
        return "success", self.sol_rpq, self.sol_slack, self.sol_lambda, self.sol_mu, self.sol_info
    else: # "x" not in dict solution
        return "x not in dict solution -- No Solution Found!!!"
    #^ if/esle
#^ solve()

def check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu):
    # returns pair (p,d) of primal/dual infeasibility (maxima)

    idx_of_trip,trip_of_idx = self.initialization(which_sources)

    n = len(trip_of_idx)
    ltrip_of_idx = n
    # Primal infeasiblility
    
    # non-negative ineqaulity
    tic_neg = time.time()
    itic_neg = time.process_time()
    max_q_negativity = 0.
    for i in range(len(self.quad_of_idx)):
        max_q_negativity = max(max_q_negativity, -sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
    #^ for
    toc_neg = time.time()
    itoc_neg = time.process_time()
    print("time to primal negativity violations 123", toc_neg - tic_neg, "secs")
    print("time to primal negativity violations 123", itoc_neg - itic_neg, "secs") 
    max_violation_of_eqn = 0.

    tic_marg = time.time()
    itic_marg = time.process_time()
    # sx** - marginals:
    for sx in self.b_sx.keys():
        mysum = self.b_sx[sx]
        for y in self.Y:
            for z in self.Z:
                s,x = sx
                if (s,x,y,z) in self.idx_of_quad.keys():
                    i = self.idx_of_quad[(s,x,y,z)]
                    q = max(0., sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
                    mysum -= q
                #^ if
            #^ for z
        #^ for y 
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ fox sx

    # s*y* - marginals:
    for sy in self.b_sy.keys():
        mysum = self.b_sy[sy]
        for x in self.X:
            for z in self.Z:
                s,y = sy
                if (s,x,y,z) in self.idx_of_quad.keys():
                    i = self.idx_of_quad[(s,x,y,z)]
                    q = max(0., sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
                    mysum -= q
                #^ if
            #^ for z
        #^ for x
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ fox sy

    # s**z - marginals:
    for sz in self.b_sz.keys():
        mysum = self.b_sz[sz]
        for x in self.X:
            for y in self.Y:
                s,z = sz
                if (s,x,y,z) in self.idx_of_quad.keys():
                    i = self.idx_of_quad[(s,x,y,z)]
                    q = max(0., sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
                    mysum -= q
                #^ if
            #^ for y
        #^ for x
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ fox sz

    toc_marg = time.time()
    itoc_marg = time.process_time()
    print("time to compute violation of marginal eqautions 123: ", toc_marg - tic_marg, "secs")
    print("time ot compute violation of marginal eqautions 123: ", itoc_marg - itic_marg, "secs") 
    primal_infeasability = max(max_violation_of_eqn, max_q_negativity)
    
    # Dual infeasiblility

    dual_infeasability = 0.
    tic_idx = time.time()
    itic_idx = time.process_time()
    idx_of_sx = dict()
    i = 0
    for s in self.S:
        for x in self.X:
            if (s,x) in self.b_sx.keys():
                idx_of_sx[(s,x)] = i
                i += 1
            #^ if sx exists
        #^ for x
    #^ for s

    idx_of_sy = dict()
    i = 0
    for s in self.S:
        for y in self.Y:
            if (s,y) in self.b_sy.keys():
                idx_of_sy[(s,y)] = i
                i += 1
            #^ if sy exists
        #^ for y
    #^ for s

    idx_of_sz = dict()
    i = 0
    for s in self.S:
        for z in self.Z:
            if (s,z) in self.b_sz.keys():
                idx_of_sz[(s,z)] = i
                i += 1
            #^ if sz exists
        #^ for z
    #^ for s
    itoc_idx = time.process_time()
    print("time to find correct dual idx 123", itoc_idx - itic_idx, "secs")
    # non-negativity dual ineqaulity
    
    itic_negD12 = time.process_time()
    if which_sources == [1,2]:

        # Get indices of dual variables of the marginal constriants
        sz_idx = defaultdict(lambda: 0.)
        for i,sxyz in enumerate(self.quad_of_idx):
            s,x,y,z = sxyz
            sz_idx[(s,x,y)] = 2*n + len(self.b_sx) + len(self.b_sy) + idx_of_sz[(s,z)]
        #^ for
        
        # Compute mu_xy: dual varaible of the q-t coupling contsraints
        mu_xy = defaultdict(lambda: 0.)
        for k, sxy in enumerate(trip_of_idx):
            s,x,y = sxy
            mu_xy[(x,y)] += sol_lambda[ n + idx_of_trip[(s,x,y)] ]
        #^ for sxy
        
        for k, sxy in enumerate(trip_of_idx):
            s,x,y = sxy
            sx_idx = 2*n + idx_of_sx[(s,x)]
            sy_idx = 2*n + len(self.b_sx) + idx_of_sy[(s,y)]
            # nu_sxy: dual variable of the q-w coupling constraints
            nu_sxy = sol_lambda[ idx_of_trip[(s,x,y)] ]
            assert nu_sxy == sol_lambda[k], "problem in nu"

            # mu_sxy: dual varaible of the q-t coupling contsraints
            mu_sxy = sol_lambda[ n + idx_of_trip[(s,x,y)] ]
            assert mu_sxy == sol_lambda[ n + k ], "problem in mu"
            

            # Find the most violated nonnegative dual ieq 
            #     a      >= 0
            dual_infeasability = max(dual_infeasability, -sol_lambda[sx_idx]
                                     - sol_lambda[sy_idx]
                                     - sol_lambda[ int(sz_idx[(s,x,y)]) ]
                                     - mu_xy[(x,y)]
                                     - nu_sxy
            )


            # Find the most violated K_exp dual ieq
            # print("dual of kexp: ", sol_lambda[sx_idx]
            #   + sol_lambda[sy_idx]
            #   + mu_xy
            #   - nu_sxy
            #   +ln(-mu_sxy)
            #   +1)
        #^ for
        itoc_negD12 = time.process_time()
        print("time to compute neagtive dual violations 12: ", itoc_negD12 - itic_negD12, "secs")
    #^ if sources
    
    itic_negD13 = time.process_time()
    if which_sources == [1,3]:

        # Get indices of dual variables of the marginal constriants
        sy_idx = defaultdict(lambda: 0.)
        for i,sxyz in enumerate(self.quad_of_idx):
            s,x,y,z = sxyz
            sy_idx[(s,x,z)] = 2*n + len(self.b_sx) + idx_of_sy[(s,z)]
        #^ for

        # mu_xz: dual varaible of the q-t coupling contsraints
        mu_xz = defaultdict(lambda: 0.)
        for k, sxz in enumerate(trip_of_idx):
            s,x,z = sxz
            mu_xz[(x,z)] += sol_lambda[ n + idx_of_trip[(s,x,z)] ]
        #^ for sxz
        
        for k, sxz in enumerate(trip_of_idx):
            s,x,z = sxz
            # Get indices of dual variables of the marginal constriants
            sx_idx = 2*n + idx_of_sx[(s,x)]
            sz_idx = 2*n + len(self.b_sx) + len(self.b_sy) + idx_of_sz[(s,z)]

            # nu_sxz: dual variable of the q-w coupling constraints
            
            nu_sxz = sol_lambda[ idx_of_trip[(s,x,z)] ]
            assert nu_sxz == sol_lambda[k], "problem in nu"

            # mu_sxz: dual varaible of the q-t coupling contsraints
            mu_sxz = sol_lambda[ n + idx_of_trip[(s,x,z)] ]
            assert mu_sxz == sol_lambda[ n + k ], "problem in mu"

            # Find the most violated nonnegative dual ieq 
            #     a      >= 0
            dual_infeasability = max(dual_infeasability, -sol_lambda[sx_idx]
                                         - sol_lambda[sy_idx[(s,x,z)]]
                                         - sol_lambda[sz_idx]
                                         - mu_xz[(x,z)]
                                         - nu_sxz
            )

            # Find the most violated K_exp dual ieq
            # print("dual of kexp: ", sol_lambda[sx_idx]
            #   + sol_lambda[sz_idx]
            #   + mu_xz
            #   - nu_sxz
            #   +ln(-mu_sxz)
            #   +1)
        #^ for
        itoc_negD13 = time.process_time()
        print("time to compute neagtive dual violations 13: ", itoc_negD13 - itic_negD13, "secs")
    #^ if sources 

    itic_negD23 = time.process_time()
    if which_sources == [2,3]:

        # Get indices of dual variables of the marginal constriants
        sx_idx = defaultdict(lambda: 0.)
        for i,sxyz in enumerate(self.quad_of_idx):
            s,x,y,z = sxyz
            sx_idx[(s,y,z)] = 2*n + idx_of_sx[(s,x)]
        #^ for

        # mu_yz: dual varaible of the q-t coupling contsraints 
        mu_yz = defaultdict(lambda: 0.)
        for k, syz in enumerate(trip_of_idx):
            s,y,z = syz
            mu_yz[(y,z)] += sol_lambda[ n + idx_of_trip[(s,y,z)] ]
        #^ for syz
        
        for k, syz in enumerate(trip_of_idx):
            s,y,z = syz
            # Get indices of dual variables of the marginal constriants
            sy_idx = 2*n + len(self.b_sx) + idx_of_sy[(s,y)]
            sz_idx = 2*n + len(self.b_sx) + len(self.b_sy) + idx_of_sz[(s,z)]

            # nu_syz: dual variable of the q-w coupling constraints
            
            nu_syz = sol_lambda[idx_of_trip[(s,y,z)]]
            assert nu_syz == sol_lambda[k], "problem in nu"

            # mu_sxy: dual varaible of the q-t coupling contsraints
            mu_syz = sol_lambda[ n + idx_of_trip[(s,y,z)] ]
            assert mu_syz == sol_lambda[ n + k ], "problem in mu"
            
            # Find the most violated nonnegative dual ieq 
            #     a      >= 0
            dual_infeasability = max(dual_infeasability, -sol_lambda[sx_idx[(s,y,z)]]
                                         - sol_lambda[sy_idx]
                                         - sol_lambda[sz_idx]
                                         - mu_yz[(y,z)]
                                         - nu_syz
            )

            # Find the most violated K_exp dual ieq
            # print("dual of kexp: ", sol_lambda[sy_idx]
            #   + sol_lambda[sz_idx]
            #   + mu_yz
            #   - nu_syz
            #   +ln(-mu_syz)
            #   +1)
        #^ for
        itoc_negD23 = time.process_time()
        print("time to compute neagtive dual violations 23: ", itoc_negD23 - itic_negD23, "secs")
    #^ if sources
    
    return primal_infeasability, dual_infeasability
#^ check_feasibility()    

def dual_value(self, sol_lambda, b):
    return -np.dot(sol_lambda, b)

def marginal_ab(self,_A, _B, _C, _D, which_sources, sol_rpq):
    # provide the positive marginals all (a,b) in a system  (a,b,c,d)
    marg_12 = dict()
    marg_13 = dict()
    marg_14 = dict()
    marg_23 = dict()
    marg_24 = dict()
    marg_34 = dict()
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    ltrip_of_idx = len(trip_of_idx)
    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if (a,b) in marg_12.keys():
            marg_12[(a,b)] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_12[(a,b)] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for 

    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if (a,c) in marg_13.keys():
            marg_13[(a,c)] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_13[(a,c)] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for 

    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if (a,d) in marg_14.keys():
            marg_14[(a,d)] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_14[(a,d)] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for 

    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if (b,c) in marg_23.keys():
            marg_23[(b,c)] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_23[(b,c)] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for

    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if (b,d) in marg_24.keys():
            marg_24[(b,d)] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_24[(b,d)] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for

    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if (c,d) in marg_34.keys():
            marg_34[(c,d)] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_34[(c,d)] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if 
    #^ for
    return marg_12, marg_13, marg_14, marg_23, marg_24, marg_34
#^ marginal_ab()

def marginal_abc(self, _A, _B, _C, _D, which_sources, sol_rpq):
    # provide the positive marginals all (a,b,c) in a system  (a,b,c,d)
    marg = dict()
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    ltrip_of_idx = len(trip_of_idx)
    for a in _A:
        for b in _B:
            for c in _C:
                for d in _D:
                    if which_sources == [1,2]: w = (a,b,c,d)
                    elif which_sources == [1,3]: w = (a,b,d,c)
                    elif which_sources == [2,3]: w = (a,d,b,c)                    
                    if w in self.idx_of_quad.keys():
                        if (a,b,c) in marg.keys():
                            marg[(a,b,c)] += max(0,sol_rpq[self.sq_vidx(self.idx_of_quad[ w ], ltrip_of_idx)])
                        else:
                            marg[(a,b,c)] = max(0,sol_rpq[self.sq_vidx(self.idx_of_quad[ w ], ltrip_of_idx)])
                        #^ if (a,b,c) is a key
                    #^ if w exists
                #^ for d
            #^ for c
        #^for b
    #^ for a 
    return marg
#^ marginal_abc()

def condentropy_2vars(self, which_sources, sol_rpq):
    # compute cond entropy of the distribution in self.sol_rpq
    mysum = 0.
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    if which_sources == [1,2]:
        # H( S | X, Y )
        marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ = self.marginal_ab(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        marg_SXY = self.marginal_abc(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        for s in self.S:
            for x in self.X:
                for y in self.Y:
                    if (s,x,y) in idx_of_trip.keys() and marg_XY[(x,y)] > 0 and  marg_SXY[(s,x,y)] > 0:
                        # subtract q_{sxy}*log( q_{sxy}/q_{xy} )
                        mysum -= marg_SXY[(s,x,y)]*log(marg_SXY[(s,x,y)]/marg_XY[(x,y)])    
        return mysum
    #^ if sources
    elif which_sources == [1,3]:
        # H( S | X, Z )
        marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ = self.marginal_ab(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        marg_SXZ = self.marginal_abc(self.S, self.X, self.Z, self.Y, which_sources, sol_rpq)
        for s in self.S:
            for x in self.X:
                for z in self.Z:
                    if (s,x,z) in idx_of_trip.keys() and marg_SXZ[(s,x,z)] > 0 and  marg_XZ[(x,z)] > 0:
                        # subtract q_{sxy}*log( q_{sxy}/q_{xy} )
                        mysum -= marg_SXZ[(s,x,z)]*log(marg_SXZ[(s,x,z)]/marg_XZ[(x,z)])
        return mysum
    #^ if sources
    elif which_sources == [2,3]:
        # H( S | Y, Z )
        marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ = self.marginal_ab(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        marg_SYZ = self.marginal_abc(self.S, self.Y, self.Z, self.X, which_sources, sol_rpq)
        for s in self.S:
            for y in self.Y:
                for z in self.Z:
                    if (s,y,z) in idx_of_trip.keys() and marg_SYZ[(s,y,z)] > 0 and  marg_YZ[(y,z)] > 0:
                        # subtract q_{sxy}*log( q_{sxy}/q_{xy} )
                        mysum -= marg_SYZ[(s,y,z)]*log(marg_SYZ[(s,y,z)]/marg_YZ[(y,z)])
        return mysum

    #^ if sources
#^ condentropy_2vars()

def marginal_a(self,_A, _B, _C, _D, which_sources, sol_rpq):
    # provide the positive marginals all a in a system  (a,b,c,d)
    marg_1 = dict()
    marg_2 = dict()
    marg_3 = dict()
    marg_4 = dict()
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    ltrip_of_idx = len(trip_of_idx)
    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if a in marg_1.keys():
            marg_1[a] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_1[a] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for
    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if b in marg_2.keys():
            marg_2[b] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_2[b] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for
    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if c in marg_3.keys():
            marg_3[c] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_3[c] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for
    for abcd in self.idx_of_quad.keys():
        a,b,c,d = abcd
        if d in marg_4.keys():
            marg_4[d] += max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        else:
            marg_4[d] = max(0, sol_rpq[self.sq_vidx(self.idx_of_quad[ (a,b,c,d) ], ltrip_of_idx)])
        #^ if
    #^ for
    return marg_1, marg_2, marg_3, marg_4
#^ marginal_ab()


def condentropy_1var(self,which_sources,sol_rpq):
    mysum = 0.
    
    if which_sources == [1,2]:
        # H( S | Z )
        marg_S,marg_X,marg_Y,marg_Z = self.marginal_a(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ = self.marginal_ab(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        for s in self.S:
            for z in self.Z:
                if (s,z) in self.b_sz.keys() and marg_SZ[(s,z)] > 0 and marg_Z[z] > 0:
                    # Subtract q_{s,z}*log( q_{s,z}/ q_{z} )
                    mysum -= marg_SZ[(s,z)]*log(marg_SZ[(s,z)]/marg_Z[z])
                #^ if sz exists
            #^ for z 
        #^ for z
        return mysum
    #^ if sources
    elif which_sources == [1,3]:
        # H( S | Y )
        marg_S,marg_X,marg_Y,marg_Z = self.marginal_a(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ = self.marginal_ab(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        for s in self.S:
            for y in self.Y:
                if (s,y) in self.b_sy.keys()and marg_SY[(s,y)] > 0 and marg_Y[y] > 0:
                    # Subtract q_{s,y}*log( q_{s,y}/ q_{y} )
                    mysum -= marg_SY[(s,y)]*log(marg_SY[(s,y)]/marg_Y[y])
                #^ if sy exists
            #^ for y 
        #^ for s
        return mysum
    #^ if sources
    
    elif which_sources == [2,3]:
        # H ( S | X )        
        marg_S,marg_X,marg_Y,marg_Z = self.marginal_a(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ = self.marginal_ab(self.S, self.X, self.Y, self.Z, which_sources, sol_rpq)
        for s in self.S:
            for x in self.X:
                if (s,x) in self.b_sx.keys() and marg_SX[(s,x)] > 0 and marg_X[x] > 0:
                    # Subtract q_{s,x}*log( q_{s,x}/ q_{x} )
                    mysum -= marg_SX[(s,x)]*log(marg_SX[(s,x)]/marg_X[x])
                #^ if sx exists
            #^ for y 
        #^ for s
        return mysum
    #^ if sources
#^ condentropy_1var()

def entropy_S(self,pdf):
    mysum = 0.
    for s in self.S:
        psum = 0.
        for x in self.X:
            if not (s,x) in self.b_sx: continue
            for y in self.Y:
                if not (s,y) in self.b_sy:  continue
                for z in self.Z:
                    if (s,x,y,z) in pdf.keys():
                        psum += pdf[(s,x,y,z)]
                    #^ if
                #^ for z
            #^ for y
        #^ for x
        mysum -= psum * log(psum)
    #^ for x
    return mysum
#^ entropy_S()

