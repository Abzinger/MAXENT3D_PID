"""
# TRIVARIATE_SYN.py -- Python Class
#
# Creates the optimization problem needed to compute synergy (also needed for uniqueness)
#
# The optimization problems is:
#
# min -H(S|XYZ)
#
# (c) Abdullah Makkeh, Dirk Oliver Theis
# Permission to use and modify under Apache License version 2.0
#
##########################################################################################
"""
import ecos
from scipy import sparse
import numpy as np
from numpy import linalg as LA
from collections import defaultdict
import math
import time 

# ECOS's exp cone: (r,p,q)   w/   q>0  &  exp(r/q) ≤ p/q
# Translation:     (0,1,2)   w/   2>0  &  0/2      ≤ ln(1/2)
def r_vidx(i):
    return 3*i
def p_vidx(i):
    return 3*i+1
def q_vidx(i):
    return 3*i+2
ln  = math.log
log = math.log2

def create_model(self, output = 0):
    """Creates the exponential Cone Program min_{q in Delta_d}H(T|X,Y,Z)
           of the form 
              min. c'x
              s.t.
                  Ax = b
                  Gx <=_K h
             where x = (r,p,q)
                   K represents a vector representing cones (K_1, K_2)
                   such that K_1 is a vector repesenting exponential cones 
                             K_2 is a vector repesenting nonnegative cones 
        
        Args:
             output:
        Returns: 
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
        
    """
    
    tic_all = time.process_time()
    n = len(self.quad_of_idx)
    m = len(self.b_tx) + len(self.b_ty) + len(self.b_tz)
    n_vars = 3*n
    n_cons = n+m
    
    # Create the equations: Ax = b
    self.b = np.zeros((n_cons,),dtype=np.double)
    
    Eqn   = []
    Var   = []
    Coeff = []
    
    # The q-p coupling equations: q_{*xyz} - p_{sxyz} = 0
    # ( Expensive step )
    itic_p = time.process_time()
    
    Eqn_dict = defaultdict(lambda: 0.)
    Eqn_dict_num = defaultdict(int)
    Eqn_dict_acc = defaultdict(list)
    Var_dict = defaultdict(list)
    Coeff_dict = defaultdict(list)

    for i,uxyz in enumerate(self.quad_of_idx):
        u,x,y,z = uxyz
        q_var = q_vidx(self.idx_of_quad[ (u,x,y,z) ])
        Var_dict[x,y,z].append( q_var )
        Coeff_dict[x,y,z].append( +1. )
        Eqn_dict[u,x,y,z] = i
        Eqn_dict_num[x,y,z] += 1
    #^ for uxyz exists

    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        temp = [ Eqn_dict[s,x,y,z] ] * Eqn_dict_num[x,y,z]
        Eqn_dict_acc[s,x,y,z] += temp
    #^ for sxyz exits
    
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        p_var   = p_vidx(i)
        Eqn.append( Eqn_dict[s,x,y,z] )
        Var.append( p_var )
        Coeff.append( -1. )
        Eqn += Eqn_dict_acc[s,x,y,z]
        Var += Var_dict[x,y,z]
        Coeff += Coeff_dict[x,y,z]
    #^ for sxyz exists
    
    itoc_p = time.process_time()
    if output == 2: print("TRIVARIATE_SYN.create_model(): Time to create q-p coupling equations [min - H(S|X,Y,V)]:", itoc_p - itic_p, "secs")

    # running number
    eqn = -1 + len(self.quad_of_idx) 

    # The marginal constraints 
    itic_m = time.process_time()   

    # The sx marginals q_{sx**} = b^x_{sx}
    Eqn_marg = defaultdict(lambda: 0.)
    Eqn_marg_num = defaultdict(int)
    Eqn_marg_acc = defaultdict(list)
    Var_marg = defaultdict(list)
    Coeff_marg = defaultdict(list)
    for sx,i in self.b_tx.items():
        (s,x) = sx
        eqn += 1
        Eqn_marg[s,x] = eqn
    #^ for sx exists
        
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        if (s,x) in self.b_tx.keys():
            q_var = q_vidx(self.idx_of_quad[ (s,x,y,z) ])
            Var_marg[s,x].append(q_var)
            Coeff_marg[s,x].append(+1.)
            Eqn_marg_num[s,x] += 1
        #^ if sx exits
    # for sxyz
    
    for sx,i in self.b_tx.items():
        (s,x) = sx
        temp = [ Eqn_marg[s,x] ]*Eqn_marg_num[s,x]
        Eqn_marg_acc[s,x] += temp
    #^ for  sx exists    
    for sx,i in self.b_tx.items():
        (s,x) = sx
        Eqn += Eqn_marg_acc[s,x]
        Var += Var_marg[s,x]
        Coeff += Coeff_marg[s,x]
        self.b[ Eqn_marg[s,x] ] = self.b_tx[(s,x)]
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
        Eqn_marg[s,y] = eqn
    #^ for sy exists
        
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        if (s,y) in self.b_ty.keys():
            q_var = q_vidx(self.idx_of_quad[ (s,x,y,z) ])
            Var_marg[s,y].append(q_var)
            Coeff_marg[s,y].append(+1.)
            Eqn_marg_num[s,y] += 1
        #^ if sy exits
    # for sxyz
    
    for sy,i in self.b_ty.items():
        (s,y) = sy
        temp = [ Eqn_marg[s,y] ]*Eqn_marg_num[s,y]
        Eqn_marg_acc[s,y] += temp
    #^ for  sy exists   
        
    for sy,i in self.b_ty.items():
        (s,y) = sy
        Eqn += Eqn_marg_acc[s,y]
        Var += Var_marg[s,y]
        Coeff += Coeff_marg[s,y]
        self.b[ Eqn_marg[s,y] ] = self.b_ty[(s,y)]
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
        Eqn_marg[s,z] = eqn
    #^ for sz exists
    
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        if (s,z) in self.b_tz.keys():
            q_var = q_vidx(self.idx_of_quad[ (s,x,y,z) ])
            Var_marg[s,z].append(q_var)
            Coeff_marg[s,z].append(+1.)
            Eqn_marg_num[s,z] += 1
        #^ if sz exits
    # for sxyz
    
    for sz,i in self.b_tz.items():
        (s,z) = sz
        temp = [ Eqn_marg[s,z] ]*Eqn_marg_num[s,z]
        Eqn_marg_acc[s,z] += temp
    #^ for  sz exists    
        
    for sz,i in self.b_tz.items():
        (s,z) = sz
        Eqn += Eqn_marg_acc[s,z]
        Var += Var_marg[s,z]
        Coeff += Coeff_marg[s,z]
        self.b[ Eqn_marg[s,z] ] = self.b_tz[(s,z)]
    #^ for sz exists

    itoc_m = time.process_time()
    if output == 2: print("TRIVARIATE_SYN.create_model(): Time to create marginal equations [min - H(S|X,Y,Z)]:", itoc_m - itic_m, "secs")

    # Store A 
    self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(n_cons,n_vars), dtype=np.double)
    
    # Generalized ieqs: gen.nneg of the variable quadruple (r_i,q_i,p_i), i=0,dots,n-1:
    Ieq   = []
    Var   = []
    Coeff = []
    for i,sxyz in enumerate(self.quad_of_idx):
        r_var = r_vidx(i)
        q_var = q_vidx(i)
        p_var = p_vidx(i)
        
        Ieq.append( len(Ieq) )
        Var.append( r_var )
        Coeff.append( -1. )
        
        Ieq.append( len(Ieq) )
        Var.append( p_var )
        Coeff.append( -1. )
        
        Ieq.append( len(Ieq) )
        Var.append( q_var )
        Coeff.append( -1. )
    #^ for sxyz

    self.G         = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(n_vars,n_vars), dtype=np.double)
    self.h         = np.zeros( (n_vars,),dtype=np.double )
    self.dims = dict()
    self.dims['e'] = n
    
    # Objective function:
    self.c = np.zeros( (n_vars,),dtype=np.double )
    for i,sxyz in enumerate(self.quad_of_idx):
        self.c[ r_vidx(i) ] = -1.
    #^ for xyz
    toc_all = time.process_time()
    if output > 0: print("TRIVARIATE_SYN.create_model(): Time to create model [min - H(S|X,Y,Z)]:", toc_all - tic_all, "secs") 
    return self.c, self.G, self.h, self.dims, self.A, self.b
#^ create_model()



def solve(self, c, G, h, dims, A, b, output):
    """ Solves the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
        
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
            sol_slack:  numpy.array - slack of primal optimal solution 
                                      (G*sol_rpq - h)
            sol_lambda: numpy.array - equalities dual optimal solution
            sol_mu:     numpy.array - inequalities dual  optimal solution   
            sol_info:   dictionary - Brief stats of the optimization from ECOS

    """

    itic = time.process_time()
    self.marg_xyz = None # for cond[]mutinf computation below
    
    if self.verbose != None:
        self.ecos_kwargs["verbose"] = self.verbose
    #^ if    
    solution = ecos.solve(c, G, h, dims,  A, b, **self.ecos_kwargs)

    if 'x' in solution.keys():
        self.sol_rpq    = solution['x']
        self.sol_slack  = solution['s']
        self.sol_lambda = solution['y']
        self.sol_mu     = solution['z']
        self.sol_info   = solution['info']
        itoc = time.process_time()
        if output == 2: print("TRIVARIATE_SYN.solve():Time to solve the Exponential Program of H(S|X,Y,Z):", itoc - itic, "secs") 
        return "success", self.sol_rpq, self.sol_slack, self.sol_lambda, self.sol_mu, self.sol_info
    else: # "x" not in dict solution
        return "x not in dict solution -- No Solution Found!!!"
    #^ if/esle
#^ solve()


def condentropy(self, sol_rpq, output = 0):
    """ evalutes the value of H(T|X,Y,Z) at the optimal distribution
        
        Args:
             sol_rpq:    numpy.array - primal optimal solution
             output: int - print different outputs based on (int) to console
        
        Returns: 
            mysum: float - H(T|X,Y,Z)

    """
    
    # compute cond entropy of the distribution in self.sol_rpq
    
    itic = time.process_time()
    marg_s = defaultdict(lambda: 0.)
    mysum = 0.
    for sxyz,i in self.idx_of_quad.items():
        s,x,y,z = sxyz
        marg_s[(x,y,z)] += sol_rpq[q_vidx(i)]
    #^ for 

    for sxyz,i in self.idx_of_quad.items():
        s,x,y,z = sxyz
        q_sxyz = sol_rpq[q_vidx(i)]
        if marg_s[(x,y,z)] > 0 and q_sxyz > 0:
            mysum -= q_sxyz*log(q_sxyz/marg_s[(x,y,z)])
        #^ if
    #^ for
    itoc = time.process_time()
    # print("H(T|XYZ)", mysum)

    if output == 2: print("TRIVARIATE_SYN.condentropy(): Time to compute condent H(S|XYZ):", itoc - itic, "secs")

    return mysum
#^ condentropy()

def check_feasibility(self, sol_rpq, sol_lambda, output = 0):
    """ Checks the KKT conditions of the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
        
        Args:
             sol_rpq:    numpy.array - primal optimal solution
             sol_slack:  numpy.array - slack of primal optimal solution 
                                      (G*sol_rpq - h)
             sol_lambda: numpy.array - equalities dual optimal solution
             output: int - print different outputs based on (int) to console

        Returns: 
             primal_infeasability: float - maximum violation of the optimal primal solution
                                           for primal equalities and inequalities
             dual_infeasability:   float - maximum violation of the optimal dual solution
                                           for dual equalities and inequalities
        
    """
    
    # returns pair (p,d) of primal/dual infeasibility (maxima)

    # Primal infeasiblility
    
    # non-negative ineqaulity
    itic_neg = time.process_time()
    max_q_negativity = 0.
    for i in range(len(self.quad_of_idx)):
        max_q_negativity = max(max_q_negativity, -sol_rpq[q_vidx(i)])
    #^ for
    itoc_neg = time.process_time()
    if output == 2: print("TRIVARIATE_SYN.check_feasibility(): Time to compute primal negative violations [min - H(S|X,Y,Z)]: ", itoc_neg - itic_neg, "secs")  


    # Marginal equations
    # ( Expensive Step )
    max_violation_of_eqn = 0.
    itic_marg = time.process_time()    
    # sx** - marginals:
    sol_b_sx = defaultdict(lambda: 0.)
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        sol_b_sx[s,x] += max(0., sol_rpq[q_vidx(i)])
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
        sol_b_sy[s,y] += max(0., sol_rpq[q_vidx(i)])
    #^ for sxyz exists 
    for sy,i in self.b_ty.items():
        s,y  = sy
        mysum = i - sol_b_sy[s,y]
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    
    # s**z - marginals:
    sol_b_sz = defaultdict(lambda: 0.)
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        sol_b_sz[s,z] += max(0., sol_rpq[q_vidx(i)])
    #^ for sxyz exists 
    for sz,i in self.b_tz.items():
        s,z  = sz
        mysum = i - sol_b_sz[s,z]
        max_violation_of_eqn = max( max_violation_of_eqn, abs(mysum) )
    #^ for sz exists

    primal_infeasability = max(max_violation_of_eqn,max_q_negativity)

    itoc_marg = time.process_time()
    if output == 2: print("TRIVARIATE_SYN.check_feasibility(): Time to compute marginal violations [min - H(S|X,Y,Z)]:", itoc_marg - itic_marg, "secs") 

    # Dual infeasiblility

    # Finding dual indices 
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

    dual_infeasability = 0.

    itic_negD = time.process_time()

    # Compute mu_*xyz
    mu_xyz = defaultdict(lambda: 0.)
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz
        mu_xyz[(x,y,z)] += sol_lambda[i]
    #^ for mu_xyz

    # Dual inequalities
    for i,sxyz in enumerate(self.quad_of_idx):
        s,x,y,z = sxyz

        # Get indices of dual variables of the marginal constriants
        sx_idx = len(self.quad_of_idx) + idx_of_sx[(s,x)]
        sy_idx = len(self.quad_of_idx) + len(self.b_tx) + idx_of_sy[(s,y)]
        sz_idx = len(self.quad_of_idx) + len(self.b_tx) + len(self.b_ty) + idx_of_sz[(s,z)]
        
        # Find the most violated dual ieq
        dual_infeasability = max( dual_infeasability, - sol_lambda[sx_idx]
                                  - sol_lambda[sy_idx]
                                  - sol_lambda[sz_idx]
                                  - mu_xyz[(x,y,z)]
                                  -ln(-sol_lambda[i])
                                  - 1
        )
    #^ for
    itoc_negD = time.process_time()
    if output == 2: print("TRIVARIATE_SYN.check_feasibility(): Time to compute dual negative violations [min - H(S|X,Y,Z)]:", itoc_negD - itic_negD, "secs")
    return primal_infeasability, dual_infeasability
#^ check_feasibility()    


def dual_value(self, sol_lambda, b):
    """ evaluates the dual value of H(T|X,Y,Z)
        
        Args:
             sol_lambda: numpy.array - equalities dual optimal solution
             b: numpy.array - L.H.S. of equalities 
        
        Returns: 
            float 

    """

    return -np.dot(sol_lambda, b)
#^ dual_value()


#EOF
