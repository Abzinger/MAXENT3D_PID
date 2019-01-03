"""TRIVARIATE_UNQ.py -- Python Class

Creates the optimization problems needed to compute unique information

The optimization problems are:

min -H(S|XY), min -H(S|XZ) and min -H(S|YZ)

(c) Abdullah Makkeh, Dirk Oliver Theis
Permission to use and modify under Apache License version 2.0
"""
import ecos
from scipy import sparse
import numpy as np
from numpy import linalg as LA
from collections import defaultdict
import math
import time
from collections import defaultdict
ln  = math.log
log = math.log2

def core_initialization(_S, X_1, X_2, b_sx1, b_sx2, idx_of, of_idx):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0

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

    return 0
#^ core_initialization()


def initialization(self, which_sources):
    """Initialize the data for the triplets (T,U,V) where U,V in {X,Y,Z}
        
        Args:
             which_sources: [1,2] if sources are X and Y 
                            [1,3] if sources are X and Z
                            [2,3] if sources are Y and Z

        Returns: 
            (if [1,2] u,v=x,y|if [1,3] u,v=x,z|if [2,3] u,v=y,z|)
            dictionary

              keys: (t,u,v)
              values: their indices 

            list of (t,u,v)
    """

    idx_of_trip  = dict()
    trip_of_idx  = []

    if which_sources == [1,2]:
        core_initialization(self.T, self.X, self.Y, self.b_tx, self.b_ty, idx_of_trip, trip_of_idx)
    elif which_sources == [1,3]:
        core_initialization(self.T, self.X, self.Z, self.b_tx, self.b_tz, idx_of_trip, trip_of_idx)
    elif which_sources == [2,3]:
        core_initialization(self.T, self.Y, self.Z, self.b_ty, self.b_tz, idx_of_trip, trip_of_idx)
    else:
        print("TRIVARIATE_UNQ.initialization(): which_sources takes the values [1,2], [1,3], or [2,3]")
        exit(1)

    return idx_of_trip, trip_of_idx
#^ initialization()

# ECOS's exp cone: (r,p,w)     w/   w>0  &  exp(r/w) ≤ p/w
# Variables here:  (r,p,w)U(q)
# Translation:     (0,1,2,3)   w/   2>0  &  0/2      ≤ ln(1/2)
def sr_vidx(i):
    """Computes the index of the optimal r (r_vars) in the
           optimal solution of the Exponential Cone Programming

        Args:
             i: int

        Returns:
            int
    """

    return 3*i
def sp_vidx(i):
    """Computes the index of the optimal (p_vars) in the
           optimal solution of the Exponential Cone Programming
    
        Args:
             i: int

        Returns:
            int
    """
    
    return 3*i+1
def sw_vidx(i):
    """Computes the index of the optimal (w_vars) in the
           optimal solution of the Exponential Cone Programming

        Args:
             i: int

        Returns:
            int
    """
    
    return 3*i+2
def sq_vidx(self, i, ltrip_of_idx):
    """Computes the index of the optimal distribution (q_vars) in the
           optimal solution of the Exponential Cone Programming

        Args:
             i: int
             ltrip_of_idx: int - length of triplet t,u,v 
        Returns:
            int
    """
    
    return 3*ltrip_of_idx + i

def create_model(self, which_sources, output):
    """Creates the exponential Cone Program min_{q in Delta_d}H(T|U,V)
           of the form 
              min. c'x
              s.t.
              Ax = b
              Gx <=_K h

           where 
           x = (r,p,q)
           K represents a vector representing cones (K_1, K_2)
           such that K_1 is a vector repesenting exponential cones 
           K_2 is a vector repesenting nonnegative cones 
        
        Args:
             which_sources: [1,2] if sources are X and Y 
                            [1,3] if sources are X and Z
                            [2,3] if sources are Y and Z

        Returns: 
            numpy.array - objective function weights
            scipy.sparse.csc_matrix - matrix of exponential and nonnegative 
            inequalities
            
            numpy.array - L.H.S. of inequalities 
            dictionary -  cones to be used 

                keys: string - cone type (exponential or nonegative)
                values: int - number of cones

            scipy.sparse.csc_matrix - Matrix of marginal, q-w coupling, and 
            q-p coupling equations 

            numpy.array - L.H.S. of equalities 
        
    """

    # Initialize which sources for the model
    tic_all = time.process_time()
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    m = len(self.b_tx) + len(self.b_ty) + len(self.b_tz)
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

    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.create_model(): Time to create q-w coupling equations [min -H(S|X,Y)]:", toc_w - tic_w, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.create_model(): Time to create q-w coupling equations [min -H(S|X,Z)]:", toc_w - tic_w, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.create_model(): Time to create q-w coupling equations [min -H(S|Y,Z)]:", toc_w - tic_w, "secs")

    # running number
    eqn = -1 + len(trip_of_idx)
    
    # The q-p coupling equations:
    #         if Sources = X,Y  q_{*tv*} - p_{stv} = 0
    #         if Sources = X,Z  q_{*t*v} - p_{stv} = 0
    #         if Sources = Y,Z  q_{**tv} - p_{stv} = 0
    # ( Expensive step )
    tic_p = time.process_time()

    Eqn_dict = defaultdict(lambda: 0.)
    Eqn_dict_num = defaultdict(int)
    Eqn_dict_acc = defaultdict(list)
    Var_dict = defaultdict(list)
    Coeff_dict = defaultdict(list)

    if which_sources == [1,2]:
        for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            eqn     += 1
            Eqn_dict[(s,t,v)] = eqn
        #^ for 
            
        for i,utvw in enumerate(self.quad_of_idx):
            u,t,v,w = utvw
            if (t,v) in self.b_xy.keys():
                Eqn_dict_num[(t,v)] += 1
                Coeff_dict[(t,v)].append(+1.)
                q_var = self.sq_vidx(self.idx_of_quad[ (u,t,v,w) ], ltrip_of_idx)
                Var_dict[(t,v)].append(q_var)
            #^ if *xy* exists
        #^ for utvw
        for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            temp = [Eqn_dict[(s,t,v)]]*Eqn_dict_num[(t,v)]
            Eqn_dict_acc[(s,t,v)] += temp
        # for 
    elif which_sources == [1,3]:
        for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            eqn     += 1
            Eqn_dict[(s,t,v)] = eqn
        #^ for
        for i,utwv in enumerate(self.quad_of_idx):
            u,t,w,v = utwv
            if (t,v) in self.b_xz.keys():
                Eqn_dict_num[(t,v)] += 1
                Coeff_dict[(t,v)].append(+1.)
                q_var = self.sq_vidx(self.idx_of_quad[ (u,t,w,v) ], ltrip_of_idx)
                Var_dict[(t,v)].append(q_var)
            #^ if *x*z exists
        #^ for utwv

        for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            temp = [Eqn_dict[(s,t,v)]]*Eqn_dict_num[(t,v)]
            Eqn_dict_acc[(s,t,v)] += temp
        # for
    elif which_sources == [2,3]:
        for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            eqn     += 1
            Eqn_dict[(s,t,v)] = eqn
        #^ for
        for i,uwtv in enumerate(self.quad_of_idx):
            u,w,t,v = uwtv
            if (t,v) in self.b_yz.keys():
                Eqn_dict_num[(t,v)] += 1
                Coeff_dict[(t,v)].append(+1.)
                q_var = self.sq_vidx(self.idx_of_quad[ (u,w,t,v) ], ltrip_of_idx)
                Var_dict[(t,v)].append(q_var)
            #^ if **yz exists
        #^ for utwv

        for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            temp = [Eqn_dict[(s,t,v)]]*Eqn_dict_num[(t,v)]
            Eqn_dict_acc[(s,t,v)] += temp
        # for
    #^ if which sources
    
    for i,stv in enumerate(trip_of_idx):
            s,t,v = stv
            p_var   = sp_vidx(i)
            Eqn.append( Eqn_dict[(s,t,v)] )
            Eqn += Eqn_dict_acc[(s,t,v)]
            Var.append( p_var )
            Var += Var_dict[(t,v)]
            Coeff.append( -1. )
            Coeff += Coeff_dict[(t,v)] 
    #^ for stv
    toc_p = time.process_time()
    
    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.create_model(): Time to create q-t coupling equations [min -H(S|X,Y)]:", toc_p - tic_p, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.create_model(): Time to create q-t coupling equations [min -H(S|X,Z)]:", toc_p - tic_p, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.create_model(): Time to create q-t coupling equations [min -H(S|Y,Z)]:", toc_p - tic_p, "secs")

    # Create the marginal constraints
    
    # The sx marginals q_{sx**} = b^x_{sx}
    tic_m = time.process_time()
    Eqn_marg = defaultdict(lambda: 0.)
    Eqn_marg_num = defaultdict(int)
    Eqn_marg_acc = defaultdict(list)
    Var_marg = defaultdict(list)
    Coeff_marg = defaultdict(list)

    for sx,i in self.b_tx.items():
        (s,x) = sx
        eqn += 1
        Eqn_marg[(s,x)] = eqn
    #^ for sx exists
        
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        if (s,x) in self.b_tx.keys():
            q_var = self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], ltrip_of_idx)
            Var_marg[(s,x)].append(q_var)
            Coeff_marg[(s,x)].append(+1.)
            Eqn_marg_num[(s,x)] += 1
            self.b[ Eqn_marg[(s,x)] ] = self.b_tx[(s,x)]
        #^ if sx exits
    # for sxyz
    
    for sx,i in self.b_tx.items():
        (s,x) = sx
        temp = [ Eqn_marg[(s,x)] ]*Eqn_marg_num[(s,x)]
        Eqn_marg_acc[(s,x)] += temp
    #^ for  sx exists    
    
    for sx,i in self.b_tx.items():
        (s,x) = sx
        Eqn += Eqn_marg_acc[(s,x)]
        Var += Var_marg[(s,x)]
        Coeff += Coeff_marg[(s,x)]
    #^ for sx exists

    # The sy marginals q_{s*y*} = b^y_{sy}

    Eqn_marg = defaultdict(lambda: 0.)
    Eqn_marg_num = defaultdict(int)
    Eqn_marg_acc = defaultdict(list)
    Var_marg = defaultdict(list)
    Coeff_marg = defaultdict(list)
    for sy,i in self.b_ty.items():
        (s,y) = sy
        eqn += 1
        Eqn_marg[(s,y)] = eqn
    #^ for sy exists
        
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        if (s,y) in self.b_ty.keys():
            q_var = self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], ltrip_of_idx)
            Var_marg[(s,y)].append(q_var)
            Coeff_marg[(s,y)].append(+1.)
            Eqn_marg_num[(s,y)] += 1
            self.b[ Eqn_marg[(s,y)] ] = self.b_ty[(s,y)]
        #^ if sy exits
    # for sxyz
    
    for sy,i in self.b_ty.items():
        (s,y) = sy
        temp = [ Eqn_marg[(s,y)] ]*Eqn_marg_num[(s,y)]
        Eqn_marg_acc[(s,y)] += temp
    #^ for  sy exists   
        
    for sy,i in self.b_ty.items():
        (s,y) = sy
        Eqn += Eqn_marg_acc[(s,y)]
        Var += Var_marg[(s,y)]
        Coeff += Coeff_marg[(s,y)]
    #^ for sy exists

    # The sz marginals q_{s**z} = b^z_{sz}
    Eqn_marg = defaultdict(lambda: 0.)
    Eqn_marg_num = defaultdict(int)
    Eqn_marg_acc = defaultdict(list)
    Var_marg = defaultdict(list)
    Coeff_marg = defaultdict(list)

    for sz,i in self.b_tz.items():
        (s,z) = sz
        eqn += 1
        Eqn_marg[(s,z)] = eqn
    #^ for sz exists
    
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        if (s,z) in self.b_tz.keys():
            q_var = self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], ltrip_of_idx)
            Var_marg[(s,z)].append(q_var)
            Coeff_marg[(s,z)].append(+1.)
            Eqn_marg_num[(s,z)] += 1
            self.b[ Eqn_marg[(s,z)] ] = self.b_tz[(s,z)]
        #^ if sz exits
    # for sxyz
    
    for sz,i in self.b_tz.items():
        (s,z) = sz
        temp = [ Eqn_marg[(s,z)] ]*Eqn_marg_num[(s,z)]
        Eqn_marg_acc[(s,z)] += temp
    #^ for  sz exists    
        
    for sz,i in self.b_tz.items():
        (s,z) = sz
        Eqn += Eqn_marg_acc[(s,z)]
        Var += Var_marg[(s,z)]
        Coeff += Coeff_marg[(s,z)]
    #^ for sz exists
    toc_m = time.process_time()
    
    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.create_model(): Time to create marginal equations [min -H(S|X,Y)]:", toc_m - tic_m, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.create_model(): Time to create marginal equations [min -H(S|X,Z)]:", toc_m - tic_m, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.create_model(): Time to create marginal equations [min -H(S|Y,Z)]:", toc_m - tic_m, "secs")

    # Store the constraints in A
    tic_rest = time.process_time()
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

    toc_rest = time.process_time()
    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.create_model(): Time to create the matrices [min -H(S|X,Y)]:", toc_rest - tic_rest, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.create_model(): Time to create the matrices [min -H(S|X,Z)]:", toc_rest - tic_rest, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.create_model(): Time to create the matrices [min -H(S|Y,Z)]:", toc_rest - tic_rest, "secs")
    toc_all = time.process_time()
    if output > 0:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.create_model(): Time to create model [min -H(S|X,Y)]:", toc_all - tic_all, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.create_model(): Time to create model [min -H(S|X,Z)]:", toc_all - tic_all, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.create_model(): Time to create model [min -H(S|Y,Z)]:", toc_all - tic_all, "secs")

    return self.c, self.G, self.h, self.dims, self.A, self.b

#^ create_model()

def solve(self, c, G, h, dims, A, b, output):
    """Solves the exponential Cone Program min_{Delta_p}H(T|U,V) where U,V in {X,Y,Z}
        
        Args:
            c: numpy.array - objective function weights
            G: scipy.sparse.csc_matrix - matrix of exponential and nonnegative 
            inequalities

            h: numpy.array - L.H.S. of inequalities 
            dims: dictionary -  cones to be used 

                    keys: string - cone type (exponential or nonegative)
                    values: int - number of cones

            A: scipy.sparse.csc_matrix - Matrix of marginal, q-w coupling, and 
            q-p coupling equations 

            b: numpy.array - L.H.S. of equalities 
            output: int - print different outputs based on (int) to console
 
       Returns: 
            sol_rpq:    numpy.array - primal optimal solution
            sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)
            sol_lambda: numpy.array - equalities dual optimal solution
            sol_mu:     numpy.array - inequalities dual  optimal solution   
            sol_info:   dictionary - Brief stats of the optimization from ECOS

    """

    itic = time.process_time()
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
        itoc = time.process_time()
        if output == 2: print("TRIVARIATE_UNQ.solve(): Time to solve the Exponential Program of H(S|W,T)", itoc - itic, "secs") 
        return "success", self.sol_rpq, self.sol_slack, self.sol_lambda, self.sol_mu, self.sol_info
    else: # "x" not in dict solution
        return "TRIVARIATE_UNQ.solve(): x not in dict solution -- No Solution Found!!!"
    #^ if/esle
#^ solve()

def check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu, output= 0):
    """Checks the KKT conditions of the exponential Cone Program min_{Delta_p}H(T|U,V) 
          where U,V in {X,Y,Z}
        
        Args:
             which_sources: [1,2] if sources are X and Y 
                            [1,3] if sources are X and Z
                            [2,3] if sources are Y and Z

             sol_rpq:    numpy.array - primal optimal solution
             sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)
             sol_lambda: numpy.array - equalities dual optimal solution
             sol_mu:     numpy.array - inequalities dual  optimal solution   
             output: int - print different outputs based on (int) to console

        Returns: 
             primal_infeasability: float - maximum violation of the optimal primal solution
                                           for primal equalities and inequalities
             dual_infeasability:   float - maximum violation of the optimal dual solution
                                           for dual equalities and inequalities

    """

    # returns pair (p,d) of primal/dual infeasibility (maxima)
    idx_of_trip,trip_of_idx = self.initialization(which_sources)

    n = len(trip_of_idx)
    ltrip_of_idx = n
    # Primal infeasiblility
    
    # non-negative ineqaulity
    itic_neg = time.process_time()
    max_q_negativity = 0.
    for i in range(len(self.quad_of_idx)):
        max_q_negativity = max(max_q_negativity, -sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
    #^ for
    toc_neg = time.time()
    itoc_neg = time.process_time()
    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute primal negativity violations [min -H(S|XY)]:", itoc_neg - itic_neg, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute primal negativity violations [min -H(S|XZ)]:", itoc_neg - itic_neg, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute primal negativity violations [min -H(S|YZ)]:", itoc_neg - itic_neg, "secs")
    #^ if printing
    max_violation_of_eqn = 0.

    itic_marg = time.process_time()
    # sx** - marginals:
    sol_b_sx = defaultdict(lambda: 0.)
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        sol_b_sx[s,x] += max(0., sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
    #^ for sxyz exists 
    for sx,i in self.b_tx.items():
        s,x  = sx
        mysum = i - sol_b_sx[s,x]
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ for sx exists
    
    # s*y* - marginals:
    sol_b_sy = defaultdict(lambda: 0.)
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        sol_b_sy[s,y] += max(0., sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
    #^ for sxyz exists 
    for sy,i in self.b_ty.items():
        s,y  = sy
        mysum = i - sol_b_sy[s,y]
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ for sy exists
    
    # s**z - marginals:
    sol_b_sz = defaultdict(lambda: 0.)
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        sol_b_sz[s,z] += max(0., sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
    #^ for sxyz exists 
    for sz,i in self.b_tz.items():
        s,z  = sz
        mysum = i - sol_b_sz[s,z]
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ for sz exists
    itoc_marg = time.process_time()
    
    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute violation of marginal eqautions [min -H(S|XY)]:", itoc_marg - itic_marg, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute violation of marginal eqautions [min -H(S|XZ)]:", itoc_marg - itic_marg, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute violation of marginal eqautions [min -H(S|YZ)]:", itoc_marg - itic_marg, "secs")
    #^ if printing
    
    primal_infeasability = max(max_violation_of_eqn, max_q_negativity)
    
    # Dual infeasiblility

    dual_infeasability = 0.
    tic_idx = time.time()
    itic_idx = time.process_time()
    idx_of_sx = dict()
    i = 0
    for s in self.T:
        for x in self.X:
            if (s,x) in self.b_tx.keys():
                idx_of_sx[(s,x)] = i
                i += 1
            #^ if sx exists
        #^ for x
    #^ for s

    idx_of_sy = dict()
    i = 0
    for s in self.T:
        for y in self.Y:
            if (s,y) in self.b_ty.keys():
                idx_of_sy[(s,y)] = i
                i += 1
            #^ if sy exists
        #^ for y
    #^ for s

    idx_of_sz = dict()
    i = 0
    for s in self.T:
        for z in self.Z:
            if (s,z) in self.b_tz.keys():
                idx_of_sz[(s,z)] = i
                i += 1
            #^ if sz exists
        #^ for z
    #^ for s
    itoc_idx = time.process_time()
    
    if output == 2:
        if which_sources == [1,2]: print("TRIVARIATE_UNQ.check_feasibility(): Time to find correct dual idx [min -H(S|XY)]:", itoc_idx - itic_idx, "secs")
        if which_sources == [1,3]: print("TRIVARIATE_UNQ.check_feasibility(): Time to find correct dual idx [min -H(S|XZ)]:", itoc_idx - itic_idx, "secs")
        if which_sources == [2,3]: print("TRIVARIATE_UNQ.check_feasibility(): Time to find correct dual idx [min -H(S|YZ)]:", itoc_idx - itic_idx, "secs")

    # non-negativity dual ineqaulity
    
    itic_negD12 = time.process_time()
    if which_sources == [1,2]:

        # Get indices of dual variables of the marginal constriants
        sz_idx = defaultdict(lambda: 0.)
        for i,sxyz in enumerate(self.quad_of_idx):
            s,x,y,z = sxyz
            sz_idx[(s,x,y)] = 2*n + len(self.b_tx) + len(self.b_ty) + idx_of_sz[(s,z)]
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
            sy_idx = 2*n + len(self.b_tx) + idx_of_sy[(s,y)]
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
        if output == 2: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute neagtive dual violations [min -H(S|XY)]:", itoc_negD12 - itic_negD12, "secs")
    #^ if sources
    
    itic_negD13 = time.process_time()
    if which_sources == [1,3]:

        # Get indices of dual variables of the marginal constriants
        sy_idx = defaultdict(lambda: 0.)
        for i,sxyz in enumerate(self.quad_of_idx):
            s,x,y,z = sxyz
            sy_idx[(s,x,z)] = 2*n + len(self.b_tx) + idx_of_sy[(s,y)]
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
            sz_idx = 2*n + len(self.b_tx) + len(self.b_ty) + idx_of_sz[(s,z)]

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
        if output == 2 : print("TRIVARIATE_UNQ.check_feasibility(): Time to compute neagtive dual violations [min -H(S|XZ)]:", itoc_negD13 - itic_negD13, "secs")
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
            sy_idx = 2*n + len(self.b_tx) + idx_of_sy[(s,y)]
            sz_idx = 2*n + len(self.b_tx) + len(self.b_ty) + idx_of_sz[(s,z)]

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
        if output == 2: print("TRIVARIATE_UNQ.check_feasibility(): Time to compute neagtive dual violations [min -H(S|YZ)]:", itoc_negD23 - itic_negD23, "secs")
    #^ if sources
     
    return primal_infeasability, dual_infeasability
#^ check_feasibility()    

def dual_value(self, sol_lambda, b):
    """Evaluates the dual value of H(T|U,V) where U,V in {X,Y,Z}
        
        Args:
             sol_lambda: numpy.array - equalities dual optimal solution
             b: numpy.array - L.H.S. of equalities 
        
        Returns: 
            float 

    """

    return -np.dot(sol_lambda, b)

def marginals(self, which_sources, sol_rpq, output):
    """Computes all the marginal distributions of the optimal distribution

        Args:
             which_sources: [1,2] if sources are X and Y and Q is the optimal
                            distribution of min_{Delta_P} H(T|X,Y)
                            [1,3] if sources are X and Z and Q is the optimal
                            distribution of min_{Delta_P} H(T|X,Z)
                            [2,3] if sources are Y and Z and Q is the optimal
                            distribution of min_{Delta_P} H(T|Y,Z)

             sol_rpq: numpy.array - array of triplets (r,p,q) of Exponential cone
             where q is the optimal distribution

             output: int - print different outputs based on (int) to console
             
        Returns: 
            dictionary - optimal marginal distribution of T
              keys: t
              values: Q(t)
            dictionary - optimal marginal distribution of X
              keys: x
              values: Q(x)
            dictionary - optimal marginal distribution of Y
              keys: y
              values: Q(y)
            dictionary - optimal marginal distribution of Z
              keys: z
              values: Q(z)
            dictionary - optimal marginal distribution of (T,X)
              keys: t,x
              values: Q(t,x)
            dictionary - optimal marginal distribution of (T,Y)
              keys: t,y
              values: Q(t,y)
            dictionary - optimal marginal distribution of (T,Z)
              keys: t,z
              values: Q(t,z)
            dictionary - optimal marginal distribution of (X,Y)
              keys: x,y
              values: Q(x,y)
            dictionary - optimal marginal distribution of (X,Z)
              keys: x,z
              values: Q(x,z)
            dictionary - optimal marginal distribution of (Y,Z)
              keys: y,z
              values: Q(y,z)
            dictionary - optimal marginal distribution of (T,X,Y)
              keys: t,x,y
              values: Q(t,x,y)
            dictionary - optimal marginal distribution of (T,X,Z)
              keys: t,x,z
              values: Q(t,x,z)
            dictionary - optimal marginal distribution of (T,Y,Z)
              keys: t,y,z
              values: Q(t,y,z)
        
    """
    
    # provide the positive marginals all of a pdf for random varibles (A,B,C,D)
    
    itic = time.process_time()

    # First order marginals
    marg_S = defaultdict(float)
    marg_X = defaultdict(float)
    marg_Y = defaultdict(float)
    marg_Z = defaultdict(float)

    # Second order marginals 
    marg_SX = defaultdict(float)
    marg_SY = defaultdict(float)
    marg_SZ = defaultdict(float)
    marg_XY = defaultdict(float)
    marg_XZ = defaultdict(float)
    marg_YZ = defaultdict(float)

    # Third order marginals
    marg_SXY = defaultdict(float)
    marg_SXZ = defaultdict(float)
    marg_SYZ = defaultdict(float)

    # Initialize the triplet 
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    ltrip_of_idx = len(trip_of_idx)

    # Compute the marginals 
    for sxyz,i in self.idx_of_quad.items():
        s,x,y,z = sxyz
        marg_S[s] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_X[x] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_Y[y] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_Z[z] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_SX[(s,x)] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_SY[(s,y)] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_SZ[(s,z)] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_XY[(x,y)] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_XZ[(x,z)] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_YZ[(y,z)] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_SXY[(s,x,y)] += max(0,sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_SXZ[(s,x,z)] += max(0,sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
        marg_SYZ[(s,y,z)] += max(0,sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
    #^ for
    itoc = time.process_time()
    if output == 2:
        if which_sources == [1,2]:
            print("TRIVARIATE_UNQ.marginals(): Time to compute marginals() of H(S|XY):", itoc - itic, "secs")
        elif which_sources == [1,3]:
            print("TRIVARIATE_UNQ.marginals(): Time to compute marginals() of H(S|XZ):", itoc - itic, "secs")
        elif which_sources == [2,3]:
            print("TRIVARIATE_UNQ.marginals(): Time to compute marginals() of H(S|YZ):", itoc - itic, "secs")
        #^ if sources
    #^ if printing
    
    return marg_S, marg_X, marg_Y, marg_Z, marg_SX, marg_SY, marg_SZ, marg_XY, marg_XZ, marg_YZ, marg_SXY, marg_SXZ, marg_SYZ

#^ marginals()


def condentropy_2vars(self, which_sources, sol_rpq, output, marg_XY, marg_XZ, marg_YZ, marg_SXY, marg_SXZ, marg_SYZ):
    """Evalutes the value of H(T|U,V) w.r.t. the optimal distribution where U,V in {X,Y,Z}
        
        Args:
             which_sources: [1,2] if sources are X and Y 
                            [1,3] if sources are X and Z
                            [2,3] if sources are Y and Z

             sol_rpq: numpy.array - primal optimal solution
             output:  int - print different outputs based on (int) to console
             marg_XY: dictionary - optimal marginal distribution of (X,Y)

                      keys: x,y
                      values: Q(x,y)

             marg_XZ: dictionary - optimal marginal distribution of (X,Z)

                      keys: x,z
                      values: Q(x,z)

             marg_YZ: dictionary - optimal marginal distribution of (Y,Z)

                      keys: y,z
                      values: Q(y,z)
             
             marg_TXY: dictionary - optimal marginal distribution of (T,X,Y)

                       keys: t,x,y
                       values: Q(t,x,y)
             
             marg_TXZ: dictionary - optimal marginal distribution of (T,X,Z)

                        keys: t,x,z
                        values: Q(t,x,z)

             marg_TYZ: dictionary - optimal marginal distribution of (T,Y,Z)

                       keys: t,y,z
                       values: Q(t,y,z)

        Returns: 
            (if [1,2]: U,V=X,Y | if [1,3]: U,V=X,Z | if [2,3]: U,V=Y,Z)
            mysum: float - H(T|U,V)

    """

    # compute cond entropy of the distribution in self.sol_rpq
    
    mysum = 0.
    idx_of_trip,trip_of_idx = self.initialization(which_sources)
    if which_sources == [1,2]:
        itic = time.process_time()
        # H( S | X, Y )
        for sxy in idx_of_trip.keys():
            s,x,y = sxy
            if marg_XY[(x,y)] > 0 and  marg_SXY[(s,x,y)] > 0:
                # subtract q_{sxy}*log( q_{sxy}/q_{xy} )
                mysum -= marg_SXY[(s,x,y)]*log(marg_SXY[(s,x,y)]/marg_XY[(x,y)])
            #^ if
        #^ for 
        itoc = time.process_time()

        if output == 2: print("TRIVARIATE_UNQ.condentropy_2vars(): Time to compute H(S|XY):", itoc - itic,"secs")
        # print("H(T|XY)", mysum)
        return mysum
    #^ if sources
    elif which_sources == [1,3]:
        # H( S | X, Z )
        itic = time.process_time()

        for sxz in idx_of_trip.keys():
            s,x,z = sxz
            if marg_XZ[(x,z)] > 0 and  marg_SXZ[(s,x,z)] > 0:
                # subtract q_{sxz}*log( q_{sxz}/q_{xz} )
                mysum -= marg_SXZ[(s,x,z)]*log(marg_SXZ[(s,x,z)]/marg_XZ[(x,z)])
            #^ if
        #^ for 
        itoc = time.process_time()

        if output == 2: print("TRIVARIATE_UNQ.condentropy_2vars(): Time to compute H(S|XZ):", itoc - itic,"secs")
        # print("H(T|XZ)", mysum)
        return mysum
    #^ if sources
    elif which_sources == [2,3]:
        # H( S | Y, Z )
        itic = time.process_time()

        for syz in idx_of_trip.keys():
            s,y,z = syz
            if marg_YZ[(y,z)] > 0 and  marg_SYZ[(s,y,z)] > 0:
                # subtract q_{syz}*log( q_{syz}/q_{yz} )
                mysum -= marg_SYZ[(s,y,z)]*log(marg_SYZ[(s,y,z)]/marg_YZ[(y,z)])
            #^ if
        #^ for 
        itoc = time.process_time()
        
        if output == 2: print("TRIVARIATE_UNQ.condentropy_2vars(): Time to compute H(S|YZ):", itoc - itic,"secs")
        # print("H(T|YZ)", mysum)
        return mysum
    #^ if sources
    else:
        print("TRIVARIATE_UNQ.condentropy_2vars(): which_sources takes the values [1,2], [1,3], [2,3]")
        exit(1)
    #^ if sources

#^ condentropy_2vars()



def condentropy_1var(self,which_sources,sol_rpq, marg_SX, marg_SY, marg_SZ, marg_X, marg_Y, marg_Z):
    # (c) Abdullah Makkeh, Dirk Oliver Theis
    # Permission to use and modify under Apache License version 2.0

    mysum = 0.
    if which_sources == [1,2]:
        # H( S | Z )
        itic = time.process_time()
        for s in self.T:
            for z in self.Z:
                if (s,z) in self.b_tz.keys() and marg_SZ[(s,z)] > 0 and marg_Z[z] > 0:
                    # Subtract q_{s,z}*log( q_{s,z}/ q_{z} )
                    mysum -= marg_SZ[(s,z)]*log(marg_SZ[(s,z)]/marg_Z[z])
                #^ if sz exists
            #^ for z 
        #^ for z
        itoc = time.process_time()
        print("Time to compute H(S|Z):", itoc - itic,"secs")
        return mysum
    #^ if sources
    elif which_sources == [1,3]:
        # H( S | Y )
        itic = time.process_time()
        for s in self.T:
            for y in self.Y:
                if (s,y) in self.b_ty.keys()and marg_SY[(s,y)] > 0 and marg_Y[y] > 0:
                    # Subtract q_{s,y}*log( q_{s,y}/ q_{y} )
                    mysum -= marg_SY[(s,y)]*log(marg_SY[(s,y)]/marg_Y[y])
                #^ if sy exists
            #^ for y 
        #^ for s
        itoc = time.process_time()
        print("Time to compute H(S|Y):", itoc - itic,"secs")
        return mysum
    #^ if sources
    
    elif which_sources == [2,3]:
        # H ( S | X )
        itic = time.process_time()
        for s in self.T:
            for x in self.X:
                if (s,x) in self.b_tx.keys() and marg_SX[(s,x)] > 0 and marg_X[x] > 0:
                    # Subtract q_{s,x}*log( q_{s,x}/ q_{x} )
                    mysum -= marg_SX[(s,x)]*log(marg_SX[(s,x)]/marg_X[x])
                #^ if sx exists
            #^ for y 
        #^ for s
        itoc = time.process_time()
        print("Time to compute H(S|X):", itoc - itic,"secs")
        return mysum
    #^ if sources
    else:
        print(" which sources takes the values: [1,2], [1,3], [2,3]")
        exit(1)
    #^ if sources
    
#^ condentropy_1var()

# def entropy_S(self,pdf, sol_rpq, which_sources, output):
#     # (c) Abdullah Makkeh, Dirk Oliver Theis
#     # Permission to use and modify under Apache License version 2.0
#     mysum = 0.
#     marg_s = defaultdict(lambda: 0.)
#     idx_of_trip,trip_of_idx = self.initialization(which_sources)
#     ltrip_of_idx = len(trip_of_idx)
#     for sxyz, i in self.idx_of_quad.items():
#         s,x,y,z = sxyz
#         marg_s[s] += max(0, sol_rpq[self.sq_vidx(i, ltrip_of_idx)])
#     #^ for
#     for s in self.S:
#         if marg_s[s] > 0: mysum -= marg_s[s]*log(marg_s[s])

#     print("H(S) new:", mysum)
#     itic = time.process_time()
#     mysum = 0.
#     for s in self.S:
#         psum = 0.
#         for x in self.X:
#             if not (s,x) in self.b_sx: continue
#             for y in self.Y:
#                 if not (s,y) in self.b_sy:  continue
#                 for z in self.Z:
#                     if (s,x,y,z) in pdf.keys():
#                         psum += pdf[(s,x,y,z)]
#                     #^ if
#                 #^ for z
#             #^ for y
#         #^ for x
#         mysum -= psum * log(psum)
#     #^ for x
#     print("H(S) old:", mysum)
#     itoc = time.process_time()
#     if output == 2: print("Time to compute H(S):", itoc - itic,"secs")
#     return mysum

# #^ entropy_S()

#EOF
