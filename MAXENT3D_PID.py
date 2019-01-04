"""MAXENT3D_PID.py -- Python module

MAXENT3D_PID: Trivariate Partial Information Decomposition via Maximum Entropy 

https://github.com/Abzinger/MAXENT3D_PID 

(c) Abdullah Makkeh, Dirk Oliver Theis

Permission to use and modify with proper attribution (Apache License version 2.0)

Information about the algorithm and examples are here:

@Article{?????????????,

          author =       {Makkeh, Abdullah and Theis, Dirk Oliver and Vicente, Raul and Chicharro, Daniel},

          title =        {????????},

          journal =      {????????},

          year =         ????,

          volume =    {??},

          number =    {?},

          pages =     {???}
}

Please cite this paper when you use this software (cf. README.md)
"""

import TRIVARIATE_SYN
import TRIVARIATE_UNQ
import TRIVARIATE_QP

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

class MAXENT3D_PID_Exception(Exception):
    """Prints exception when MAXENT3D_PID doesn't return a solution 
    """
    pass


class Solve_w_ECOS():

    """Solve_w_Ecos.
    
    (c) Abdullah Makkeh, Dirk Oliver Theis

    Permission to use and modify under Apache License version 2.0

    Implements the ecos initialization and functions needed to be used by the children classes that computes PID terms  
    
    Methods:
      condentropy__orig(pdf, output)
               Computes H(T|X,Y,Z) w.r.t the original distribution P
      entropy_V(V,pdf,output)
          Computes H(T), H(X), H(Y), or H(Z) w.r.t the original distribution P
    """
    def __init__(self, marg_tx, marg_ty, marg_tz, marg_xy, marg_xz, marg_yz):

        """
        (c) Abdullah Makkeh, Dirk Oliver Theis

        Permission to use and modify under Apache License version 2.0
        
        Args: 
            marg_tx: dict() P(T,X)

            marg_ty: dict() P(T,Y)

            marg_tz: dict() P(T,Z)

            marg_xy: dict() P(X,Y)

            marg_xz: dict() P(X,Z)

            marg_yz: dict() P(Y,Z)
        """
        # ECOS parameters
        self.ecos_kwargs   = dict()
        self.verbose       = False

        # Probability density funciton data
        self.b_tx         = dict(marg_tx)
        self.b_ty         = dict(marg_ty)
        self.b_tz         = dict(marg_tz)

        self.b_xy         = dict(marg_xy)
        self.b_xz         = dict(marg_xz)
        self.b_yz         = dict(marg_yz)

        self.T            =set([ t for t,x in self.b_tx.keys() ]
                               + [ t for t,y in self.b_ty.keys() ]
                               + [ t for t,z in self.b_tz.keys() ])
        self.X            = set( [ x  for t,x in self.b_tx.keys() ] )
        self.Y            = set( [ y  for t,y in self.b_ty.keys() ] )
        self.Z            = set( [ z  for t,z in self.b_tz.keys() ] )
        self.idx_of_quad  = dict()
        self.quad_of_idx  = []

        # Do stuff:
        for t in self.T:
            for x in self.X:
                if (t,x) in self.b_tx.keys():
                    for y in self.Y:
                        if (t,y) in self.b_ty.keys():
                            for z in self.Z:
                                if (t,z) in self.b_tz.keys():
                                    self.idx_of_quad[ (t,x,y,z) ] = len( self.quad_of_idx )
                                    self.quad_of_idx.append( (t,x,y,z) )
                                #^ if
                            #^ for z
                        #^ if
                    #^ for y
                #^ if
            #^ for x
        #^ for t
    #^ init()

    def condentropy__orig(self,pdf,output):

        """Computes H(T|X,Y,Z) w.r.t. the original distribution P of (T,X,Y,Z)
        
        Args:
             pdf: dictionary - the original distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)

             output: int - print different outputs based on (int) to console

        Returns: 
            mysum: float - H(T|X,Y,Z)        
        """
        
        itic = time.process_time()
        mysum = 0.
        marg_xyz = defaultdict(lambda: 0.)
        for txyz, i in pdf.items():
            t,x,y,z = txyz
            marg_xyz[x,y,z] += pdf[(t,x,y,z)]
        #^ for
        
        for txyz, i in pdf.items():
            t,x,y,z = txyz
            p = pdf[(t,x,y,z)]
            mysum -= p*log(p/marg_xyz[x,y,z])
        #^ for
        itoc = time.process_time()
        if output == 2: print("MAXENT3D_PID.condentropy__orig(): Time to compute H(T|XYZ) of the input pdf:", itoc - itic, "secs")
        return mysum
    #^ condentropy__orig()

    # Compute H(T), H(X), H(Y), or H(Z)
    def entropy_V(self, V, pdf, output):
        """Computes H(T), H(X), H(Y), or H(Z) w.r.t. the original 
           distribution P of (T,X,Y,Z)
        
        Args:
             V: int -  
                       if 1 computes  H(T) 

                       if 2 computes  H(X) 

                       if 3 computes  H(Y) 

                       if 4 computes  H(Z) 

             pdf: dictionary - the input distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)

             output: int - print different outputs based on (int) to console

        Returns: 
            (if 1: V=T | if 2: V=X | if 3: V=Y | if 4: V=Z)

            mysum: float - H(V)
        """

        marg_V = defaultdict(lambda: 0.)
        num_V  = defaultdict(lambda: 0.) 
        mysum = 0.
        if V == 1:
            # H(S)
            itic = time.process_time()
            for txyz,r in pdf.items():
                t,x,y,z = txyz
                if r > 0:
                    marg_V[t] += r
                    num_V[t] = 1
                #^ if 
            #^ for

            for t in num_V.keys():
                if marg_V[t] > 0: mysum -= marg_V[t]*log( marg_V[t] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("MAXENT3D_PID.entopry_V(): Time to compute H(T):", itoc - itic, "secs")
            return mysum

        elif V == 2:
            # H(X)
            itic = time.process_time()
            for txyz,r in pdf.items():
                t,x,y,z = txyz
                if r > 0:
                    marg_V[x] += r
                    num_V[x] = 1
                #^ if 
            #^ for

            for x in num_V.keys():
                mysum -= marg_V[x]*log( marg_V[x] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("MAXENT3D_PID.entropy_V(): Time to compute H(X):", itoc - itic, "secs")
            return mysum

        elif V == 3:
            # H(Y)
            itic = time.process_time()
            for txyz,r in pdf.items():
                t,x,y,z = txyz
                if r > 0:
                    marg_V[y] += r
                    num_V[y] = 1
                #^ if 
            #^ for

            for y in num_V.keys():
                mysum -= marg_V[y]*log( marg_V[y] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("MAXENT3D_PID.entropy_V(): Time to compute H(Y):", itoc - itic, "secs")
            return mysum

        elif V == 4:
            # H(Z)
            itic = time.process_time()
            for txyz,r in pdf.items():
                t,x,y,z = txyz
                if r > 0:
                    marg_V[z] += r
                    num_V[z] = 1
                #^ if 
            #^ for

            for z in num_V.keys():
                mysum -= marg_V[z]*log( marg_V[z] )
            #^ for
            itoc = time.process_time()
            if output == 2: print("MAXENT3D_PID.entropy_V(): Time to compute H(Z):", itoc - itic, "secs")
            return mysum
        else:
            print("MAXENT3D_PID.entropy_V(): The argument V takes the values: 1, 2, 3, or 4")
            exit(1)
    #^ entropy()

#^ class Solve_w_ECOS



class Opt_I(Solve_w_ECOS):

    """Implements the functions to compute Synergistic information ( CI(T;X,Y,Z )

    (c) Abdullah Makkeh, Dirk Oliver Theis

    Permission to use and modify under Apache License version 2.0

    Methods:
       create_model(output)
          Creates the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
       solve(c,G,h,dims,A,b,output)
          Solves the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
       dual_value(sol_lambda,b)
          evaluates the dual value of H(T|X,Y,Z)
       check_feasibility(sol_rpq,sol_lambda,output)
          Checks the KKT conditions of the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
       condentropy(sol_rpq,output)
          evalutes the value of H(T|X,Y,Z) at the optimal distribution
    """
    def create_model(self, output):
        """Creates the exponential Cone Program min_{q in Delta_d}H(T|X,Y,Z) of the form 

              min. c'x

              s.t.

                  Ax = b

                  Gx <=_K h

             where 

                   x = (r,p,q)

                   K represents a vector representing cones (K_1, K_2) such that K_1 is a vector repesenting exponential cones and K_2 is a vector repesenting nonnegative cones 
        
        Args:
             output: int - print different outputs based on (int) to console

        Returns: 
            c: numpy.array - objective function weights

            G: scipy.sparse.csc_matrix - matrix of exponential and nonnegative inequalities

            h: numpy.array - L.H.S. of inequalities 

            dims: dictionary -  cones to be used 

                keys: string - cone type (exponential or nonegative)
                values: int - number of cones

            A: scipy.sparse.csc_matrix - Matrix of marginal, q-w coupling, and q-p coupling equations 

            b: numpy.array - L.H.S. of equalities 
        
        """
        
        return TRIVARIATE_SYN.create_model(self, output)
    
    def solve(self, c, G, h, dims, A, b, output):
        """Solves the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
        
        Args:
            c: numpy.array - objective function weights

            G: scipy.sparse.csc_matrix - matrix of exponential and nonnegative inequalities

            h: numpy.array - L.H.S. of inequalities 

            dims: dictionary -  cones to be used 

                    keys: string - cone type (exponential or nonegative)
                    values: int - number of cones

            A: scipy.sparse.csc_matrix - Matrix of marginal, q-w coupling, and q-p coupling equations 

            b: numpy.array - L.H.S. of equalities 

            output: int - print different outputs based on (int) to console
 
       Returns: 
            sol_rpq:    numpy.array - primal optimal solution

            sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)

            sol_lambda: numpy.array - equalities dual optimal solution

            sol_mu:     numpy.array - inequalities dual  optimal solution   

            sol_info:   dictionary - Brief stats of the optimization from ECOS

        """

        return TRIVARIATE_SYN.solve(self, c, G, h, dims, A, b, output)
    
    def dual_value(self, sol_lambda, b):
        """Evaluates the dual value of H(T|X,Y,Z)
        
        Args:
             sol_lambda: numpy.array - equalities dual optimal solution

             b: numpy.array - L.H.S. of equalities 
        
        Returns: 
            float 

        """
        return TRIVARIATE_SYN.dual_value(self, sol_lambda, b)

    def check_feasibility(self, sol_rpq, sol_lambda, output):
        """Checks the KKT conditions of the exponential Cone Program min_{Delta_p}H(T|X,Y,Z)
        
        Args:
             sol_rpq:    numpy.array - primal optimal solution

             sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)

             sol_lambda: numpy.array - equalities dual optimal solution

             output: int - print different outputs based on (int) to console

        Returns: 
             primal_infeasability: float - maximum violation of the optimal primal solution for primal equalities and inequalities

             dual_infeasability:   float - maximum violation of the optimal dual solution for dual equalities and inequalities

        """
        return TRIVARIATE_SYN.check_feasibility(self, sol_rpq, sol_lambda, output)
    
    def condentropy(self, sol_rpq, output):
        """Evalutes the value of H(T|X,Y,Z) at the optimal distribution
        
        Args:
             sol_rpq: numpy.array - primal optimal solution

             output:  int - print different outputs based on (int) to console
        
        Returns: 
            mysum: float - H(T|X,Y,Z)

        """
        return TRIVARIATE_SYN.condentropy(self, sol_rpq, output)
#^ subclass Opt_I

# Subclass to compute Unique Information 
class Opt_II(Solve_w_ECOS):

    """Implements the functions to compute Unique information

    (c) Abdullah Makkeh, Dirk Oliver Theis

    Permission to use and modify under Apache License version 2.0
    
    Methods:
       initialization(which_sources)
          Initialize the data for the triplets (T,U,V) where U,V in {X,Y,Z}
       sq_vidx(i,which_sources)
          Computes the index of the optimal distribution (q_vars) in sol_rpq
       marginals(which_sources,sol_rpq,output)
          Computes all the marginal distributions of the optimal distribution
       create_model(which_sources,output)
          Creates the exponential Cone Program min_{Delta_p}H(T|U,V) where U,V in {X,Y,Z}
       solve(c,G,h,dims,A,b,output)
          Solves the exponential Cone Program min_{Delta_p}H(T|U,V) where U,V in {X,Y,Z}
       dual_value(sol_lambda,b)
          Evaluates the dual value of H(T|U,V) where U,V in {X,Y,Z}
       check_feasibility(which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu, output)
          Checks the KKT conditions of the exponential Cone Program min_{Delta_p}H(T|U,V) 
          where U,V in {X,Y,Z}
       condentropy_2vars(which_sources,sol_rpq,output,marg_XY,marg_XZ,marg_YZ,marg_SXY,marg_SXZ,marg_SYZ)
          Evalutes the value of H(T|U,V) w.r.t. the optimal distribution where U,V in {X,Y,Z}

    """
    
    def initialization(self, which_sources):
        """Initialize the data for the triplets (T,V,W) where V,W in {X,Y,Z}
        
        Args:
             which_sources: list(int) -
                            [1,2] if sources are X and Y 

                            [1,3] if sources are X and Z

                            [2,3] if sources are Y and Z

        Returns: 
            (if [1,2] v,w=x,y|if [1,3] v,w=x,z|if [2,3] v,w=y,z|)

            dictionary

              keys: (t,v,w)
              values: their indices 

            list of (t,v,w)
        """
        return TRIVARIATE_UNQ.initialization(self, which_sources)

    def sq_vidx(self, i, which_sources):
        """Computes the index of the optimal distribution (q_vars) in the optimal solution of the Exponential Cone Programming

        Args:
             i: int

             which_sources: list(int) -
                            [1,2] if sources are X and Y 

                            [1,3] if sources are X and Z

                            [2,3] if sources are Y and Z
        Returns:
            int
        """
        return TRIVARIATE_UNQ.sq_vidx(self, i, which_sources)

    def marginals(self, which_sources, sol_rpq, output):
        """Computes all the marginal distributions of the optimal distribution

        Args:
             which_sources: list(int) -
                            [1,2] if sources are X and Y and Q is the optimal distribution of min_{Delta_P} H(T|X,Y)
                            [1,3] if sources are X and Z and Q is the optimal distribution of min_{Delta_P} H(T|X,Z)
                            [2,3] if sources are Y and Z and Q is the optimal distribution of min_{Delta_P} H(T|Y,Z)

             sol_rpq: numpy.array - array of triplets (r,p,q) of Exponential cone where q is the optimal distribution

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

        return TRIVARIATE_UNQ.marginals(self, which_sources,sol_rpq, output)

    def create_model(self, which_sources, output):
        """Creates the exponential Cone Program min_{q in Delta_d}H(T|U,V) of the form 

              min. c'x

              s.t.

                  Ax = b

                  Gx <=_K h

             where 

                   x = (r,p,q)

                   K represents a vector representing cones (K_1, K_2) such that K_1 is a vector repesenting exponential cones and K_2 is a vector repesenting nonnegative cones 
        
        Args:
             which_sources: list(int) -
                            [1,2] if sources are X and Y 

                            [1,3] if sources are X and Z

                            [2,3] if sources are Y and Z

        Returns: 
            numpy.array - objective function weights

            scipy.sparse.csc_matrix - matrix of exponential and nonnegative inequalities

            numpy.array - L.H.S. of inequalities 

            dictionary -  cones to be used 

                keys: string - cone type (exponential or nonegative)
                values: int - number of cones

            scipy.sparse.csc_matrix - Matrix of marginal, q-w coupling, and q-p coupling equations 

            numpy.array - L.H.S. of equalities 
        
        """

        return TRIVARIATE_UNQ.create_model(self, which_sources, output)
    
    def solve(self, c, G, h, dims, A, b, output):
        """Solves the exponential Cone Program min_{Delta_p}H(T|U,V) where U,V in {X,Y,Z}
        
        Args:
            c: numpy.array - objective function weights

            G: scipy.sparse.csc_matrix - matrix of exponential and nonnegative inequalities

            h: numpy.array - L.H.S. of inequalities 

            dims: dictionary -  cones to be used 

                    keys: string - cone type (exponential or nonegative)
                    values: int - number of cones

            A: scipy.sparse.csc_matrix - Matrix of marginal, q-w coupling, and q-p coupling equations 

            b: numpy.array - L.H.S. of equalities 

            output: int - print different outputs based on (int) to console
 
       Returns: 

            sol_rpq:    numpy.array - primal optimal solution

            sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)

            sol_lambda: numpy.array - equalities dual optimal solution

            sol_mu:     numpy.array - inequalities dual  optimal solution   

            sol_info:   dictionary - Brief stats of the optimization from ECOS

        """

        return TRIVARIATE_UNQ.solve(self, c, G, h, dims, A, b, output)
    
    def dual_value(self, sol_lambda, b):
        """Evaluates the dual value of H(T|U,V) where U,V in {X,Y,Z}
        
        Args:
             sol_lambda: numpy.array - equalities dual optimal solution

             b: numpy.array - L.H.S. of equalities 
        
        Returns: 
            float 

        """

        return TRIVARIATE_UNQ.dual_value(self, sol_lambda, b)

    def check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu, output):
        """Checks the KKT conditions of the exponential Cone Program min_{Delta_p}H(T|U,V) where U,V in {X,Y,Z}
        
        Args:
             which_sources: list(int) - 
                            [1,2] if sources are X and Y 

                            [1,3] if sources are X and Z

                            [2,3] if sources are Y and Z

             sol_rpq:    numpy.array - primal optimal solution

             sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)

             sol_lambda: numpy.array - equalities dual optimal solution

             sol_mu:     numpy.array - inequalities dual  optimal solution   

             output: int - print different outputs based on (int) to console

        Returns: 

             primal_infeasability: float - maximum violation of the optimal primal solution for primal equalities and inequalities

             dual_infeasability:   float - maximum violation of the optimal dual solution for dual equalities and inequalities

        """

        return TRIVARIATE_UNQ.check_feasibility(self, which_sources, sol_rpq, sol_slack, sol_lambda, sol_mu, output)
    
    def condentropy_2vars(self, which_sources,sol_rpq,output,marg_XY,marg_XZ,marg_YZ,marg_TXY,marg_TXZ,marg_TYZ):
        """Evalutes the value of H(T|U,V) w.r.t. the optimal distribution where U,V in {X,Y,Z}
        
        Args:
             which_sources: list(int) - 
                            [1,2] if sources are X and Y 

                            [1,3] if sources are X and Z

                            [2,3] if sources are Y and Z

             sol_rpq:    numpy.array - primal optimal solution

             output: int - print different outputs based on (int) to console

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

        return TRIVARIATE_UNQ.condentropy_2vars(self, which_sources,sol_rpq,output,marg_XY,marg_XZ,marg_YZ,marg_TXY,marg_TXZ,marg_TYZ)

#^ subclass Opt_II

class QP():
    """Implements the functions that recover the solution (if needed) using Quadratic Programming

    (c) Abdullah Makkeh, Dirk Oliver Theis

    Permission to use and modify under Apache License version 2.0
    
    Methods:
       create_model(which_sources,output)
          Creates the second-order cone program min_{Pi_x} (x - y)^2  
       solve(c,G,h,dims,A,b,output)
          Solves the second-order cone program min_{Pi_x} (x - y)^2  
    """
    
    def __init__(self, CI, SI, UIX, UIY, UIZ, UIXY, UIXZ, UIYZ, MI, MIX, MIY, MIZ):
        """
        (c) Abdullah Makkeh, Dirk Oliver Theis

        Permission to use and modify under Apache License version 2.0
        
        Args: 
            CI:(float,float) - returned synergy and confidence of the returned synergy

            SI:(float,float) - returned shared and confidence of the returned shared

            UIX:(float,float) - returned unique X and confidence of the returned unique X

            UIY:(float,float) - returned unique Y and confidence of the returned unique Y

            UIZ:(float,float) - returned unique Z and confidence of the returned unique Z

            UIXY:(float,float) - returned unique X Y and confidence of the returned unique X Y

            UIXZ:(float,float) - returned unique X Z and confidence of the returned unique X Z

            UIYZ:(float,float) - returned unique Y Z and confidence of the returned unique Y Z
            
            MI: float - MI(T;X,Y,Z)

            MIX:float - MI(T;X)

            MIY:float - MI(T;Y)

            MIZ:float - MI(T;Z)

        where the confidence is computed based on the duality gaps of the failed optimization problems 
        """

        # ECOS parameters
        self.ecos_kwargs   = dict()
        self.verbose       = False

        # Probability density funciton data
        self.CI_e, self.CI_c     = CI
        self.SI_e, self.SI_c     = SI

        self.UIX_e, self.UIX_c   = UIX
        self.UIY_e, self.UIY_c   = UIY
        self.UIZ_e, self.UIZ_c   = UIZ

        self.UIXY_e, self.UIXY_c = UIXY
        self.UIXZ_e, self.UIXZ_c = UIXZ
        self.UIYZ_e, self.UIYZ_c = UIYZ

        self.MI  = MI
        self.MIX = MIX
        self.MIY = MIY
        self.MIZ = MIZ
        self.dims = dict()
    #^ init()

    def create_model(self, which_probs):
        """Creates the second-order cone program min_{Pi_x}1/2 x^TWx + f^T x of the form 

                      min 1/2 x^TWx + f^T x

                      subject to

                            Ax  = b

                            x >= 0

           The model can be written as SOCP:

           min 1/2 t + f^T x

           subject to

                 Ax = b

                [W^(1/2), t, 1] in SOC

                 x >= 0

           In ECOS 

               min c^T[t,x]

               subject to

                     Ax = b

                     [-I, W^(1/2)]*[t,x]^T <_{K} [0, 0, 0]

           where K := R_+ . Q, c^T := [1/2,f^T]

        Args:
             which_probs: list(int) - 
                          [1]  if min_{Delta_p}H(T|X,Y,Z) failed

                          [12] if min_{Delta_p}H(T|X,Y) failed

                          [13] if min_{Delta_p}H(T|X,Z) failed

                          [23] if min_{Delta_p}H(T|Y,Z) failed

                          [1,12]  if min_{Delta_p}H(T|X,Y,Z) and min_{Delta_p}H(T|X,Y) failed

                          [1,13]  if min_{Delta_p}H(T|X,Y,Z) and min_{Delta_p}H(T|X,Z) failed

                          [1,23]  if min_{Delta_p}H(T|X,Y,Z) and min_{Delta_p}H(T|Y,Z) failed

                          [12,13] if min_{Delta_p}H(T|X,Y) and min_{Delta_p}H(T|X,Z) failed

                          [12,23] if min_{Delta_p}H(T|X,Y) and min_{Delta_p}H(T|Y,Z) failed

                          [13,23] if min_{Delta_p}H(T|X,Z) and min_{Delta_p}H(T|Y,Z) failed

                          [1,12,13]  if min_{Delta_p}H(T|X,Y,Z), min_{Delta_p}H(T|X,Y), and min_{Delta_p}H(T|X,Z) failed

                          [1,12,23]  if min_{Delta_p}H(T|X,Y,Z), min_{Delta_p}H(T|X,Y), and min_{Delta_p}H(T|Y,Z) failed

                          [1,13,23]  if min_{Delta_p}H(T|X,Y,Z), min_{Delta_p}H(T|X,Z), and min_{Delta_p}H(T|Y,Z) failed

                          [12,13,23] if min_{Delta_p}H(T|X,Y), min_{Delta_p}H(T|X,Z), and min_{Delta_p}H(T|Y,Z) failed

        Returns: 
            numpy.array - objective function weights

            scipy.sparse.csc_matrix - matrix of soc and nonnegative inequalities

            numpy.array - L.H.S. of inequalities 

            dictionary -  cones to be used 

                keys: string - cone type (soc or nonegative)
                values: int - number of cones

            scipy.sparse.csc_matrix - Matrix of identity equations 

            numpy.array - L.H.S. of equalities 
        
        """
        
        return TRIVARIATE_QP.create_model(self, which_probs)
    
    def solve(self, c, G, h, dims, A, b, output):
        """Solves the second-order cone program min_{Pi_x}1/2 x^TWx + f^T x
        
        Args:
            c: numpy.array - objective function weights

            G: scipy.sparse.csc_matrix - matrix of soc and nonnegative inequalities

            h: numpy.array - L.H.S. of inequalities 

            dims: dictionary -  cones to be used 

                    keys: string - cone type (soc or nonegative)
                    values: int - number of cones

            A: scipy.sparse.csc_matrix - Matrix of identity equations 

            b: numpy.array - L.H.S. of equalities 

            output: int - print different outputs based on (int) to console
 
       Returns: 

            sol_tx:     numpy.array - primal optimal solution

            sol_slack:  numpy.array - slack of primal optimal solution (G*sol_rpq - h)

            sol_lambda: numpy.array - equalities dual optimal solution

            sol_mu:     numpy.array - inequalities dual  optimal solution   

            sol_info:   dictionary - Brief stats of the optimization from ECOS

        """
        
        return TRIVARIATE_QP.solve(self, c, G, h, dims, A, b, output)

#^ class QP

# Compute Marginals

def marginal_tx(p):
    """Computes the original marginal distribution of T and X 

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
       Args: 
            p: dictionary - original distribution of (T,X,Y,Z)
                 keys: (t,x,y,z)
                 values: P(t,x,y,z)

       Returns: 
            dictionary - original marginal distribution of T and X
              keys: (t,x)
              values: P(t,x)
    
    """
    
    marg = dict()
    for txyz,r in p.items():
        t,x,y,z = txyz
        if (t,x) in marg.keys():    marg[(t,x)] += r
        else:                       marg[(t,x)] =  r
    return marg
#^ marginal_tx()

def marginal_ty(p):
    """Computes the original marginal distribution of T and Y 

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
       Args: 
            p: dictionary - original distribution of (T,X,Y,Z)
                 keys: (t,x,y,z)
                 values: P(t,x,y,z)

       Returns: 
            dictionary - original marginal distribution of T and Y
              keys: (t,y)
              values: P(t,y)

    """
    
    marg = dict()
    for txyz,r in p.items():
        t,x,y,z = txyz
        if (t,y) in marg.keys():   marg[(t,y)] += r
        else:                      marg[(t,y)] =  r
    return marg
#^ marginal_ty()

def marginal_tz(p):
    """Computes the original marginal distribution of T and Z

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
       Args: 
            p: dictionary - original distribution of (T,X,Y,Z)
                 keys: (t,x,y,z)
                 values: P(t,x,y,z)

       Returns: 
            dictionary - original marginal distribution of T and Z
              keys: (t,z)
              values: P(t,z)

    """
    
    marg = dict()
    for txyz,r in p.items():
        t,x,y,z = txyz
        if (t,z) in marg.keys():   marg[(t,z)] += r
        else:                      marg[(t,z)] =  r
    return marg
#^ marginal_tz()

def marginal_xy(p):
    """Computes the original marginal distribution of X and Y 

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
       Args: 
            p: dictionary - original distribution of (T,X,Y,Z) 
                 keys: (t,x,y,z)
                 values: P(t,x,y,z)

       Returns: 
            dictionary - original marginal distribution of X and Y
              keys: (x,y)
              values: P(x,y)

    """
    
    marg = dict()
    for txyz,r in p.items():
        t,x,y,z = txyz
        if (x,y) in marg.keys():   marg[(x,y)] += r
        else:                      marg[(x,y)] =  r
    return marg
#^ marginal_xy()

def marginal_xz(p):
    """Computes the original marginal distribution of X and Z 

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
       Args: 
            p: dictionary - original distribution of (T,X,Y,Z) 
                 keys: (t,x,y,z)
                 values: P(t,x,y,z)

       Returns: 
            dictionary - original marginal distribution of X and Z
              keys: (x,z)
              values: P(x,z)

    """
    
    marg = dict()
    for txyz,r in p.items():
        t,x,y,z = txyz
        if (x,z) in marg.keys():   marg[(x,z)] += r
        else:                      marg[(x,z)] =  r
    return marg
#^ marginal_xz()

def marginal_yz(p):
    """Computes the original marginal distribution of Y and Z 

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
       Args: 
            p: dictionary - original distribution of (T,X,Y,Z) 
                 keys: (t,x,y,z)
                 values: P(t,x,y,z)

       Returns: 
            dictionary - original marginal distribution of Y and Z
              keys: (y,z)
              values: P(y,z)

    """
    
    marg = dict()
    for txyz,r in p.items():
        t,x,y,z = txyz
        if (y,z) in marg.keys():   marg[(y,z)] += r
        else:                      marg[(y,z)] =  r
    return marg
#^ marginal_yz()


# Compute Conditional Entopy of the form H(T|V)
def condent_V(V, p, output = 0):
    """Computes H(T|X), H(T|Y), or H(T|Z) w.r.t. the original distribution P of (T,X,Y,Z)

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
        Args:
             V: int -  
                     if 1 computes  H(T|X) 
                     if 2 computes  H(T|Y) 
                     if 3 computes  H(T|Z) 

             pdf: dictionary - the input distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)

             output: int - print different outputs based on (int) to console (default = 0)

        Returns: 
            (if 1: V=X | if 2: V=Y | if 3: V=Z)

            mysum: float - H(T|V)
    """
    

    # Initialization
    marg_V = defaultdict(lambda: 0.)
    mysum = 0.
    
    # Compute H(T|X)
    if V == 1:
        itic = time.process_time()
        b_tx = marginal_tx(p)

        # Get P(*,x,*,*)
        for txyz,r in p.items():
            t,x,y,z = txyz
            if r > 0: marg_V[x] += r
        #^ for txyz exists

        # Subtract P(t,x,*,*) * ( P(t,x,*,*) / P(*,x,*,*) )  
        for tx,r in b_tx.items():
            t,x = tx
            if r > 0: mysum -= r*log(r/marg_V[x])
        #^ for tx exists
        itoc = time.process_time()
        if output == 2: print("MAXENT3D_PID.condent_V(): Time to compute H(T|X):", itoc - itic, "secs")
        return mysum
    #^ if

    # Compute H(T|Y)
    if V == 2:
        itic = time.process_time()
        b_ty = marginal_ty(p)

        # Get P(*,*,y,*)
        for txyz,r in p.items():
            t,x,y,z = txyz
            if r > 0: marg_V[y] += r
        #^ for txyz exists
        
        # Subtract P(s,*,y,*) * ( P(s,*,y,*) / P(*,*,y,*) )  
        for ty,r in b_ty.items():
            t,y = ty
            if r > 0: mysum -= r*log( r / marg_V[y] )
        #^ for ty exists
        itoc = time.process_time()
        if output ==2: print("MAXENT3D_PID.condent_V(): Time to compute H(T|Y):", itoc - itic, "secs")
        return mysum
    #^ if

    # Compute H(T|Z)
    if V == 3:
        itic = time.process_time()
        b_tz = marginal_tz(p)

        # Get P(*,*,*,z)
        for txyz,r in p.items():
            t,x,y,z = txyz
            if r > 0: marg_V[z] += r
        #^ for txyz exists

        # Subtract P(t,*,*,z) * ( P(t,*,*,z) / P(*,*,*,z) )  
        for tz,r in b_tz.items():
            t,z = tz
            if r > 0: mysum -= r*log( r / marg_V[z] )
        #^ for tz exists
        itoc = time.process_time()
        if output == 2: print("MAXENT3D_PID.condent_V(): Time to compute H(T|Z):", itoc - itic, "secs")
    return mysum
    #^ if
#^ condent_V()

# Compute the Mutual Information MI(T,V)
def I_V(V,p):
    """Computes MI(T;X), MI(T;Y), or MI(T;Z) w.r.t. the original distribution P of (T,X,Y,Z)

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
        Args:
             V: int -  
                       if 1 computes  MI(T;X) 

                       if 2 computes  MI(T;Y) 

                       if 3 computes  MI(T;Z) 

             p: dictionary - the input distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)

        Returns: 
            (if 1: V=X | if 2: V=Y | if 3: V=Z)

            mysum: float - MI(T;V)
    """
    
    # Initialization
    marg_V = defaultdict(lambda: 0.)
    marg_T = defaultdict(lambda: 0.)
    mysum = 0.

    # Compute I(T; X)
    if V == 1:
        b_tx = marginal_tx(p)
        for txyz,r in p.items():
            t,x,y,z = txyz
            # Get P(t,*,*,*) & P(*,x,*,*)
            if r > 0:
                marg_T[t] += r                
                marg_V[x] += r
        #^ for txyz exists

        # add P(t,x,*,*)*( P(t,x,*,*) / (P(t,*,*,*) * P(*,x,*,*) )
        for tx,r in b_tx.items():
            t,x = tx
            if r > 0: mysum += r*log( r / ( marg_T[t]*marg_V[x] ) )
        #^ for tx exists
    #^ if

    # Compute I(T; Y)
    if V == 2:
        b_ty = marginal_ty(p)
        for txyz,r in p.items():
            t,x,y,z = txyz
            # Get P(t,*,*,*) & P(*,*,y,*)
            if r > 0:
                marg_T[t] += r                
                marg_V[y] += r
        #^ for txyz exists

        # add P(t,*,y,*)*( P(t,*,y,*) / (P(t,*,*,*) * P(*,*,y,*) )
        for ty,r in b_ty.items():
            t,y = ty
            if r > 0: mysum += r*log( r / ( marg_T[t]*marg_V[y] ) )
        #^ for ty exists
    #^ if
    
    # Compute I(T; Z)
    if V == 3:
        b_tz = marginal_tz(p)

        # Get P(t,*,*,*) & P(*,*,*,z)
        for txyz,r in p.items():
            t,x,y,z = txyz
            if r > 0:
                marg_T[t] += r                
                marg_V[z] += r
        #^ for txyz exists

        # add P(t,*,*,z)*( P(t,*,*,z) / (P(t,*,*,*) * P(*,*,*,z) ) 
        for tz,r in b_tz.items():
            t,z = tz
            if r > 0: mysum += r*log( r / (marg_T[t]*marg_V[z]) )
        #^ for tz exists
    #^ if

    return mysum
#^ I_V()



def I_VW(V,p):
    """Computes MI(T;X,Y), MI(T;X,Z), or MI(T;Y,Z) w.r.t. the original distribution P of (T,X,Y,Z)

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
        Args:
             V: int -  
                       if 12 computes  MI(T;X,Y) 

                       if 13 computes  MI(T;X,Z) 

                       if 23 computes  MI(T;Y,Z) 

             pdf: dictionary - the input distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)

             output: int - print different outputs based on (int) to console
                     (default = 0)

        Returns: 
            (if 12: V,W=X,Y | if 13: V,W=X,Z | if 23: V,W=Y,Z)

            mysum: float - MI(T;V,W)
    """
    
    # Initialization
    marg_V = defaultdict(lambda: 0.)
    marg_VV = defaultdict(lambda: 0.)
    marg_T = defaultdict(lambda: 0.)
    mysum = 0.

    # Compute I(T; XY)
    if V == 12:
        for txyz,r in p.items():
            t,x,y,z = txyz
            # Get P(t,*,*,*), P(*,x,y,*), P(t,x,y,*)
            if r > 0:
                marg_T[t] += r                
                marg_V[x,y] += r
                marg_VV[t,x,y] += r
        #^ for txyz exists

        # add P(t,x,y,*)*log( P(t,x,y,*) / (P(t,*,*,*) * P(*,x,y,*) )
        for txy,r in marg_VV.items():
            t,x,y = txy
            mysum += r*log( r / ( marg_T[t]*marg_V[x,y] ) )
        #^ for tx exists
    #^ if

    # Compute I(T; XZ)
    elif V == 13:
        for txyz,r in p.items():
            t,x,y,z = txyz
            # Get P(t,*,*,*), P(*,x,*,z) & P(t,x,*,z)
            if r > 0:
                marg_T[t] += r                
                marg_V[x,z] += r
                marg_VV[t,x,z] += r
        #^ for txyz exists

        # add P(t,x,*,z)*log( P(t,x,*,z) / (P(t,*,*,*) * P(*,x,*,z) )
        for txz,r in marg_VV.items():
            t,x,z = txz
            mysum += r*log( r / ( marg_T[t]*marg_V[x,z] ) )
        #^ for ty exists
    #^ if
    
    # Compute I(T; YZ)
    elif V == 23:
        # Get P(t,*,*,*), P(*,*,y,z) & P(t,*,y,z)
        for txyz,r in p.items():
            t,x,y,z = txyz
            if r > 0:
                marg_T[t] += r                
                marg_V[y,z] += r
                marg_VV[t,y,z] += r
                
        #^ for txyz exists

        # add P(t,*,y,z)*log( P(t,*,y,z) / (P(t,*,*,*) * P(*,*,y,z) ) 
        for tyz,r in marg_VV.items():
            t,y,z = tyz
            mysum += r*log( r / (marg_T[t]*marg_V[y,z]) )
        #^ for tz exists
    #^ if

    return mysum
#^ I_VW()


def I_XYZ(p):
    """Computes MI(T;X,Y,Z) w.r.t. the original distribution P of (T,X,Y,Z)

       (c) Abdullah Makkeh, Dirk Oliver Theis
       Permission to use and modify under Apache License version 2.0
        
        Args:
             pdf: dictionary - the original distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)
        Returns: 
            mysum: float - MI(T;X,Y,Z)
    """
    # Initialization
    marg_V = defaultdict(lambda: 0.)
    marg_T = defaultdict(lambda: 0.)
    mysum = 0.

    # Compute I(T; XYZ)
    for txyz,r in p.items():
        t,x,y,z = txyz
        # Get P(t,*,*,*), P(*,x,y,z)
        if r > 0:
            marg_T[t] += r                
            marg_V[x,y,z] += r
        #^ if 
    #^ for txyz exists

    # add P(t,x,y,z)*log( P(t,x,y,z) / (P(t,*,*,*) * P(*,x,y,z) )
    for txyz,r in p.items():
        t,x,y,z = txyz
        mysum += r*log( r / ( marg_T[t]*marg_V[x,y,z] ) )
    #^ for tx exists
    return mysum
#^ I_XYZ()


def pid(pdf_dirty, cone_solver='ECOS', output=0, parallel='off', **solver_args):
    """Computes the partial information decomposition of  (T,X,Y,Z)

       (c) Abdullah Makkeh, Dirk Oliver Theis

       Permission to use and modify under Apache License version 2.0
        
        Args:
             pdf_dirty: dictionary - the original distribution of (T,X,Y,Z)
                    Keys: (t,x,y,z)
                    values: P(t,x,y,z)
                    (dirty refers to the values = 0)
             
             cone_solver: string - name of the cone solver 
                          (Default = 'ECOS')
                          (Currently only ECOS) 

             output: int - print different outputs based on (int) to console 
                     (default = 0)
             
             parallel: string - determines whether computation will be done in parallel
                       if 'off' sequential computation 

                       if 'on'  parallel computations 

                       (default = 'off')

             **solver_args: pointer to dictionary - dictionary of ECOS parameters 
                            (default = None)

        Returns: 
                return_data: dictionary - estimated decomposition, solver used, numerical error
                         
    """

    assert type(pdf_dirty) is dict, "MAXENT3D_PID.pid(pdf): pdf must be a dictionary"
    assert type(cone_solver) is str, "MAXENT3D_PID.pid(pdf): `cone_solver' parameter must be string (e.g., 'ECOS')"
    if __debug__:
        sum_p = 0
        for k,v in pdf_dirty.items():
            assert type(k) is tuple or type(k) is list,           "MAXENT3D_PID.pid(pdf): pdf's keys must be tuples or lists"
            assert len(k)==4,                                     "MAXENT3D_PID.pid(pdf): pdf's keys must be tuples/lists of length 4"
            assert type(v) is float or ( type(v)==int and v==0 ), "MAXENT3D_PID.pid(pdf): pdf's values must be floats"
            assert v > -.1,                                       "MAXENT3D_PID.pid(pdf): pdf's values must not be negative"
            sum_p += v
        #^ for
        assert abs(sum_p - 1)< 1.e-10,                                       "MAXENT3D_PID.pid(pdf): pdf's values must sum up to 1 (tolerance of precision is 1.e-10)"
    #^ if
    assert type(output) is int, "MAXENT3D_PID.pid(pdf,output): output must be an integer"

    # Check if the solver is implemented:
    assert cone_solver=="ECOS", "MAXENT3D_PID.pid(pdf): We currently don't have an interface for the Cone Solver "+cone_solver+" (only ECOS)."

    pdf = { k:v  for k,v in pdf_dirty.items() if v > 1.e-300 }

    tic_marg = time.time()
    bx_tx = marginal_tx(pdf)
    by_ty = marginal_ty(pdf)
    bz_tz = marginal_tz(pdf)

    b_xy = marginal_xy(pdf)
    b_xz = marginal_xz(pdf)
    b_yz = marginal_yz(pdf)
    
    toc_marg = time.time()
    if output > 0: print("\nMAXENT3D_PID.pid(): Time to create marginals:", toc_marg - tic_marg, "secs\n")
    # if cone_solver=="ECOS": .....
    if output > 0:  print("\nMAXENT3D_PID.pid(): Preparing Cone Program data",end="...\n")

    solver = Solve_w_ECOS(bx_tx, by_ty, bz_tz, b_xy, b_xz, b_yz)
    subsolver_I = Opt_I(bx_tx, by_ty, bz_tz, b_xy, b_xz, b_yz)
    subsolver_II = Opt_II(bx_tx, by_ty, bz_tz, b_xy, b_xz, b_yz)

    tic_mod = time.time()
        
    if parallel == 'on':
        pool = Pool()
        # call create_model() for min -H( T|XY ), min -H( T|XZ ), min -H( T|YZ )
        cre_12 = pool.apply_async(subsolver_II.create_model, [ [1,2], output ])
        cre_13 = pool.apply_async(subsolver_II.create_model, [ [1,3], output ])
        cre_23 = pool.apply_async(subsolver_II.create_model, [ [2,3], output ])

        # call create_model() for min -H( T|XYZ )
        cre_I = pool.apply_async(subsolver_I.create_model, [ output ])

        # Get the models
        c_I, G_I, h_I, dims_I, A_I, b_I = cre_I.get()
        c_12, G_12, h_12, dims_12, A_12, b_12 = cre_12.get()
        c_13, G_13, h_13, dims_13, A_13, b_13 = cre_13.get()
        c_23, G_23, h_23, dims_23, A_23, b_23 = cre_23.get()
        pool.close()
        pool.join()
    else:
        # create model min -H( T|XYZ )
        c_I, G_I, h_I, dims_I, A_I, b_I = subsolver_I.create_model(output)

        # create models min -H( T|XY ), min -H( T|XZ ), min -H( T|YZ )
        c_12, G_12, h_12, dims_12, A_12, b_12 = subsolver_II.create_model([1,2], output)
        c_13, G_13, h_13, dims_13, A_13, b_13 = subsolver_II.create_model([1,3], output)
        c_23, G_23, h_23, dims_23, A_23, b_23 = subsolver_II.create_model([2,3], output)
    #^ if parallel
    
    toc_mod = time.time()
    if output > 0: print("\nMAXENT3D_PID.pid(): Time to create all models. ", toc_mod - tic_mod, "secs\n")

    if output > 2:
        subsolver_I.verbose = True
        subsolver_II.verbose = True
        

    ecos_keep_solver_obj = False
    if 'keep_solver_object' in solver_args.keys():
        if solver_args['keep_solver_object']==True: ecos_keep_solver_obj = True
        del solver_args['keep_solver_object']

    subsolver_I.ecos_kwargs = solver_args
    subsolver_II.ecos_kwargs = solver_args
    if output > 0: print("MAXENT3D_PID.pid(): Preparing Cone Program data is done.")

    if output == 1: print("MAXENT3D_PID.pid(): Starting solver",end="...")
    if output > 2: print("MAXENT3D_PID.pid(): Starting solver.")

    if parallel == "on":

        # Find the optimal solution of: min H(T|XYZ), min H(T|XY), H(T|XZ) and H(T|YZ)
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
        # Solve the optimization: min -H( T|XYZ )
        retval_I, sol_rpq_I, sol_slack_I, sol_lambda_I, sol_mu_I, sol_info_I = subsolver_I.solve(c_I, G_I, h_I, dims_I, A_I, b_I, output)

        # Solve the optimizations: min -H( T|XY ), min -H( T|XZ ), min -H( T|YZ )
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
            raise MAXENT3D_PID_Exception("MAXENT3D_PID_Exception: Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")
        #^ if (keep solver)
    #^ if (solve failure)

    if output > 0:  print("\nMAXENT3D_PID.pid(): Solving is done.\n")

    tic_stats = time.time()
    if output > 0:
        print("\nMAXENT3D_PID.pid(): Stats for optimizing H(T|X,Y,Z):\n", sol_info_I)
        print("\nMAXENT3D_PID.pid(): Stats for optimizing H(T|X,Y):\n", sol_info_12)
        print("\nMAXENT3D_PID.pid(): Stats for optimizing H(T|X,Z):\n", sol_info_13)
        print("\nMAXENT3D_PID.pid(): Stats for optimizing H(T|Y,Z):\n", sol_info_23)
    if parallel == 'on':

        # Compute the value of the dual objective function for each optimization:
        # min -H( T|XYZ ) min -H( T|XY ), min -H( T|XZ ) and  min -H( T|YZ ) ) 
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
        marg_12_T, marg_12_X, marg_12_Y, marg_12_Z, marg_12_TX, marg_12_TY, marg_12_TZ, marg_12_XY, marg_12_XZ, marg_12_YZ, marg_12_TXY, marg_12_TXZ, marg_12_TYZ = marg_12.get()
        marg_13_T, marg_13_X, marg_13_Y, marg_13_Z, marg_13_TX, marg_13_TY, marg_13_TZ, marg_13_XY, marg_13_XZ, marg_13_YZ, marg_13_TXY, marg_13_TXZ, marg_13_TYZ = marg_13.get()
        marg_23_T, marg_23_X, marg_23_Y, marg_23_Z, marg_23_TX, marg_23_TY, marg_23_TZ, marg_23_XY, marg_23_XZ, marg_23_YZ, marg_23_TXY, marg_23_TXZ, marg_23_TYZ = marg_23.get()
        pool.close()
        pool.join()


        # Compute H(T|X,Y,Z), H(T|X,Y), H(T|X,Z), H(T|Y,Z) (using the optimal pdf) 
        pool = Pool()

        # Call condentropy() and condentropy_2vars() 
        con_I = pool.apply_async(subsolver_I.condentropy,[sol_rpq_I, output])
        con_12 = pool.apply_async(subsolver_II.condentropy_2vars,[ [1,2], sol_rpq_12, output, marg_12_XY, marg_12_XZ, marg_12_YZ,marg_12_TXY, marg_12_TXZ, marg_12_TYZ])
        con_13        = pool.apply_async(subsolver_II.condentropy_2vars,[[1,3], sol_rpq_13, output, marg_13_XY, marg_13_XZ, marg_13_YZ, marg_13_TXY, marg_13_TXZ, marg_13_TYZ])
        con_23        = pool.apply_async(subsolver_II.condentropy_2vars,[[2,3], sol_rpq_23, output, marg_23_XY, marg_23_XZ, marg_23_YZ, marg_23_TXY, marg_23_TXZ, marg_23_TYZ])

        # Get H(T|X,Y,Z), H(T|X,Y), H(T|X,Z) and H(T|Y,Z)
        condent_I     = con_I.get()
        condent_12    = con_12.get()
        condent_13    = con_13.get()
        condent_23    = con_23.get()

        pool.close()
        pool.join()

        # Compute H(T), H(T|X), H(T|Y) and H(T|Z) 
        pool = Pool()

        # Call entropy_V() and condent_V()
        ent_T = pool.apply_async(solver.entropy_V, [1, pdf, output])
        con_1 = pool.apply_async(condent_V, [1, pdf, output])
        con_2 = pool.apply_async(condent_V, [2, pdf, output])
        con_3 = pool.apply_async(condent_V, [3, pdf, output])

        # Get H(T), H(T|X), H(T|Y) and H(T|Z) 
        entropy_T     = ent_T.get()

        condent_1     = con_1.get()
        condent_2     = con_2.get()
        condent_3     = con_3.get()
        pool.close()
        pool.join()

        # Compute H(T|X,Y,Z) of the original pdf (not optimal pdf)
        condent__orig = solver.condentropy__orig(pdf, output)
    else:

        # Compute the value of the dual objective function for each optimization:
        # min -H( T|XYZ ) min -H( T|XY ), min -H( T|XZ ) and  min -H( T|YZ )
        dual_val_I    = subsolver_I.dual_value(sol_lambda_I, b_I)
        dual_val_12   = subsolver_II.dual_value(sol_lambda_12, b_12)
        dual_val_13   = subsolver_II.dual_value(sol_lambda_13, b_13)
        dual_val_23   = subsolver_II.dual_value(sol_lambda_23, b_23)

        # Compute the marginals of the optimal pdf (part of the optimal solution is a pdf) 
        marg_12_T, marg_12_X, marg_12_Y, marg_12_Z, marg_12_TX, marg_12_TY, marg_12_TZ, marg_12_XY, marg_12_XZ, marg_12_YZ, marg_12_TXY, marg_12_TXZ, marg_12_TYZ = subsolver_II.marginals([1,2], sol_rpq_12, output)
        
        marg_13_T, marg_13_X, marg_13_Y, marg_13_Z, marg_13_TX, marg_13_TY, marg_13_TZ, marg_13_XY, marg_13_XZ, marg_13_YZ, marg_13_TXY, marg_13_TXZ, marg_13_TYZ = subsolver_II.marginals([1,3], sol_rpq_13, output)

        marg_23_T, marg_23_X, marg_23_Y, marg_23_Z, marg_23_TX, marg_23_TY, marg_23_TZ, marg_23_XY, marg_23_XZ, marg_23_YZ, marg_23_TXY, marg_23_TXZ, marg_23_TYZ = subsolver_II.marginals([2,3], sol_rpq_23, output)


        # Compute H(T|X,Y,Z), H(T|X,Y), H(T|X,Z), H(T|Y,Z) (using the optimal pdf) 
        condent_I     = subsolver_I.condentropy(sol_rpq_I, output)
        condent_12    = subsolver_II.condentropy_2vars([1,2], sol_rpq_12, output, marg_12_XY, marg_12_XZ, marg_12_YZ,marg_12_TXY, marg_12_TXZ, marg_12_TYZ)
        condent_13    = subsolver_II.condentropy_2vars([1,3],sol_rpq_13, output, marg_13_XY, marg_13_XZ, marg_13_YZ,marg_13_TXY, marg_13_TXZ, marg_13_TYZ)
        condent_23    = subsolver_II.condentropy_2vars([2,3], sol_rpq_23, output, marg_23_XY, marg_23_XZ, marg_23_YZ,marg_23_TXY, marg_23_TXZ, marg_23_TYZ)

        # Compute H(T), H(T|X), H(T|Y) and H(T|Z) 
        entropy_T     = solver.entropy_V(1, pdf, output)
        condent_1     = condent_V(1, pdf, output)
        condent_2     = condent_V(2, pdf, output)
        condent_3     = condent_V(3, pdf, output)

        # Compute H(T|X,Y,Z) of the original pdf (not optimal pdf) 
        condent__orig = solver.condentropy__orig(pdf, output)
    # elsif cone_solver=="SCS":
    # .....
    # #^endif
    toc_stats = time.time()
    if output > 0: print("\nMAXENT3D_PID.pid(): Time for retrieving results:", toc_stats - tic_stats, "secs\n")


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
    return_data["SI"]    = entropy_T  - condent__orig - return_data["CI"] - return_data["UIX"]  - return_data["UIY"]  - return_data["UIZ"] - return_data["UIXY"] - return_data["UIXZ"] - return_data["UIYZ"]

    tic_o = time.time()    
    if parallel == "on":

        # Compute the feasibility violations of the optimization problems:
        # min -H( T|XYZ ) min -H( T|XY ), min -H( T|XZ ) and  min -H( T|YZ ) ) 
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
        # min -H( T|XYZ ) min -H( T|XY ), min -H( T|XZ ) and  min -H( T|YZ ) ) 

        primal_infeas_I,dual_infeas_I = subsolver_I.check_feasibility(sol_rpq_I,sol_lambda_I,output)
        

        primal_infeas_12,dual_infeas_12 = subsolver_II.check_feasibility([1,2], sol_rpq_12, sol_slack_12, sol_lambda_12, sol_mu_12,output)
        

        primal_infeas_13,dual_infeas_13 = subsolver_II.check_feasibility([1,3], sol_rpq_13, sol_slack_13, sol_lambda_13, sol_mu_13,output)
        

        primal_infeas_23,dual_infeas_23 = subsolver_II.check_feasibility([2,3], sol_rpq_23, sol_slack_23, sol_lambda_23, sol_mu_23,output)
    #^if parallel

    toc_o = time.time()
    if output > 0: print("\nMAXENT3D_PID.pid(): Time for computing Numerical Errors:", toc_o - tic_o, "secs\n")

    # Store the numerical violations of the optimization problems:
    # min -H( T|XYZ ) min -H( T|XY ), min -H( T|XZ ) and  min -H( T|YZ ) )
    
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
    if output > 0: print("\nMAXENT3D_PID.pid(): Time for storing results:", toc_dict - tic_dict, "secs\n")

    # Recovering Solutions if duality is violated
    which_probs = []
    gap = 0.
    gap_I = 0.
    gap_12 = 0.
    gap_13 = 0.
    gap_23 = 0.

    # Determine which problems didn't converge
    if return_data["Num_err_I"][2] >= 1.e-4:
        print("MAXENT3D_PID.pid(): Numerical problems [min - H(T|XYZ)]")
        which_probs.append(1)
        gap_I = return_data["Num_err_I"][2]
    if return_data["Num_err_12"][2] >= 1.e-4:
        print("MAXENT3D_PID.pid(): Numerical problems [min - H(T|XY)]")
        which_probs.append(12)
        gap_12 = return_data["Num_err_12"][2]
    if return_data["Num_err_13"][2] >= 1.e-4:
        print("MAXENT3D_PID.pid(): Numerical problems [min - H(T|XZ)]")
        which_probs.append(13)
        gap_13 = return_data["Num_err_13"][2]
    if return_data["Num_err_23"][2] >= 1.e-4:
        print("MAXENT3D_PID.pid(): Numerical problems [min - H(T|YZ)]")
        which_probs.append(23)
        gap_23 = return_data["Num_err_23"][2]
    if len(which_probs) == 4:
        which_probs = []
    #^ if doesn't converge
    itic_recover = time.process_time()
    # Launch the approperiate QP to recover
    # if which_probs == []:
    #     continue 
    QP_solver_args = dict()
    QP_solver_args['max_iters'] = 1000
    # QP_solver_args['keep_solver_object'] = True
    QP_solver_args['abstol']= 1.e-15
    QP_solver_args['reltol']= 1.e-15
    QP_solver_args['feastol']= 1.e-15
    if which_probs == [1]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        CI   = ( return_data["CI"],   1.-gap_I )
        SI   = ( return_data["SI"],   1.-gap_I )
        UIX  = ( return_data["UIX"],  1.-gap_I )
        UIY  = ( return_data["UIY"],  1.-gap_I )
        UIZ  = ( return_data["UIZ"],  1.-gap_I )
        UIXY = ( return_data["UIXY"], 1.-gap_I )
        UIXZ = ( return_data["UIXZ"], 1.-gap_I )
        UIYZ = ( return_data["UIYZ"], 1.-gap_I )

        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]

    elif which_probs == [12]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        CI   = ( return_data["CI"],   0.        )
        SI   = ( return_data["SI"],   1.-gap_12 )
        UIX  = ( return_data["UIX"],  0.        )
        UIY  = ( return_data["UIY"],  0.        )
        UIZ  = ( return_data["UIZ"],  1.-gap_12 )
        UIXY = ( return_data["UIXY"], 0.        )
        UIXZ = ( return_data["UIXZ"], 1.-gap_12 )
        UIYZ = ( return_data["UIYZ"], 1.-gap_12 )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIZ"]  = sol_tx[2]
        return_data["UIXZ"] = sol_tx[3]
        return_data["UIYZ"] = sol_tx[4]

    elif which_probs == [13]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        CI   = ( return_data["CI"],   0.        )
        SI   = ( return_data["SI"],   1.-gap_13 )
        UIX  = ( return_data["UIX"],  0.        )
        UIY  = ( return_data["UIY"],  1.-gap_13 )
        UIZ  = ( return_data["UIZ"],  0.        )
        UIXY = ( return_data["UIXY"], 1.-gap_13 )
        UIXZ = ( return_data["UIXZ"], 0.        )
        UIYZ = ( return_data["UIYZ"], 1.-gap_13 )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIY"]  = sol_tx[2]
        return_data["UIXY"] = sol_tx[3]
        return_data["UIYZ"] = sol_tx[4]

    elif which_probs == [23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        CI   = ( return_data["CI"],   0.        )
        SI   = ( return_data["SI"],   1.-gap_23 )
        UIX  = ( return_data["UIX"],  1.-gap_23 )
        UIY  = ( return_data["UIY"],  0.        )
        UIZ  = ( return_data["UIZ"],  0.        )
        UIXY = ( return_data["UIXY"], 1.-gap_23 )
        UIXZ = ( return_data["UIXZ"], 1.-gap_23 )
        UIYZ = ( return_data["UIYZ"], 0.        )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIX"]  = sol_tx[2]
        return_data["UIXY"] = sol_tx[3]
        return_data["UIXZ"] = sol_tx[4]
        
    elif which_probs == [1,12]:
        print("MAXENT3D_PID.pid():Recovering solution by Quadratic Programming")
        gap  = gap_I + gap_12
        CI   = ( return_data["CI"],   1.-gap_I )
        SI   = ( return_data["SI"],   1.-gap   )
        UIX  = ( return_data["UIX"],  1.-gap_I )
        UIY  = ( return_data["UIY"],  1.-gap_I )
        UIZ  = ( return_data["UIZ"],  1.-gap   )
        UIXY = ( return_data["UIXY"], 1.-gap_I )
        UIXZ = ( return_data["UIXZ"], 1.-gap   )
        UIYZ = ( return_data["UIYZ"], 1.-gap   )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        

        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]        

    elif which_probs == [1,13]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_I + gap_13
        CI   = ( return_data["CI"],   1.-gap_I )
        SI   = ( return_data["SI"],   1.-gap   )
        UIX  = ( return_data["UIX"],  1.-gap_I )
        UIY  = ( return_data["UIY"],  1.-gap   )
        UIZ  = ( return_data["UIZ"],  1.-gap_I )
        UIXY = ( return_data["UIXY"], 1.-gap   )
        UIXZ = ( return_data["UIXZ"], 1.-gap_I )
        UIYZ = ( return_data["UIYZ"], 1.-gap   )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]

    elif which_probs == [1,23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_I + gap_23
        CI   = ( return_data["CI"],   1.-gap_I )
        SI   = ( return_data["SI"],   1.-gap   )
        UIX  = ( return_data["UIX"],  1.-gap   )
        UIY  = ( return_data["UIY"],  1.-gap_I )
        UIZ  = ( return_data["UIZ"],  1.-gap_I )
        UIXY = ( return_data["UIXY"], 1.-gap   )
        UIXZ = ( return_data["UIXZ"], 1.-gap   )
        UIYZ = ( return_data["UIYZ"], 1.-gap_I )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]

    elif which_probs == [12,13]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_12 + gap_13
        CI   = ( return_data["CI"],   0.        )
        SI   = ( return_data["SI"],   1.-gap    )
        UIX  = ( return_data["UIX"],  0.        )
        UIY  = ( return_data["UIY"],  1.-gap_13 )
        UIZ  = ( return_data["UIZ"],  1.-gap_12 )
        UIXY = ( return_data["UIXY"], 1.-gap_13 )
        UIXZ = ( return_data["UIXZ"], 1.-gap_12 )
        UIYZ = ( return_data["UIYZ"], 1.-gap    )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIY"]  = sol_tx[2]
        return_data["UIZ"]  = sol_tx[3]
        return_data["UIXY"] = sol_tx[4]
        return_data["UIXZ"] = sol_tx[5]
        return_data["UIYZ"] = sol_tx[6]

    elif which_probs == [12,23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_12 + gap_23
        CI   = ( return_data["CI"],   0.        )
        SI   = ( return_data["SI"],   1.-gap    )
        UIX  = ( return_data["UIX"],  1.-gap_23 )
        UIY  = ( return_data["UIY"],  0.        )
        UIZ  = ( return_data["UIZ"],  1.-gap_12 )
        UIXY = ( return_data["UIXY"], 1.-gap_23 )
        UIXZ = ( return_data["UIXZ"], 1.-gap    )
        UIYZ = ( return_data["UIYZ"], 1.-gap_12 )

        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)

        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIX"]  = sol_tx[2]
        return_data["UIZ"]  = sol_tx[3]
        return_data["UIXY"] = sol_tx[4]
        return_data["UIXZ"] = sol_tx[5]
        return_data["UIYZ"] = sol_tx[6]

    elif which_probs == [13,23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_13 + gap_23
        CI   = ( return_data["CI"],   0.        )
        SI   = ( return_data["SI"],   1.-gap    )
        UIX  = ( return_data["UIX"],  1.-gap_23 )
        UIY  = ( return_data["UIY"],  1.-gap_13 ) 
        UIZ  = ( return_data["UIZ"],  0.        )
        UIXY = ( return_data["UIXY"], 1.-gap    )
        UIXZ = ( return_data["UIXZ"], 1.-gap_23 )
        UIYZ = ( return_data["UIYZ"], 1.-gap_13 )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIX"]  = sol_tx[2]
        return_data["UIY"]  = sol_tx[3]
        return_data["UIXY"] = sol_tx[4]
        return_data["UIXZ"] = sol_tx[5]
        return_data["UIYZ"] = sol_tx[6]

    elif which_probs == [1,12,13]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_I + gap_12 + gap_13
        CI   = ( return_data["CI"],   1.-gap_I            )
        SI   = ( return_data["SI"],   1.-gap              )
        UIX  = ( return_data["UIX"],  1.-gap_I            )
        UIY  = ( return_data["UIY"],  1.-(gap_I + gap_13) )
        UIZ  = ( return_data["UIZ"],  1.-(gap_I + gap_12) ) 
        UIXY = ( return_data["UIXY"], 1.-(gap_I + gap_13) )
        UIXZ = ( return_data["UIXZ"], 1.-(gap_I + gap_12) )
        UIYZ = ( return_data["UIYZ"], 1.-gap              )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]

    elif which_probs == [1,12,23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_I + gap_12 + gap_23
        CI   = ( return_data["CI"],   1.-gap_I            )
        SI   = ( return_data["SI"],   1.-gap              )
        UIX  = ( return_data["UIX"],  1.-(gap_I + gap_23) )
        UIY  = ( return_data["UIY"],  1.-gap_I            )
        UIZ  = ( return_data["UIZ"],  1.-(gap_I + gap_12) )
        UIXY = ( return_data["UIXY"], 1.-(gap_I + gap_23) )
        UIXZ = ( return_data["UIXZ"], 1.-gap              )
        UIYZ = ( return_data["UIYZ"], 1.-(gap_I + gap_12) )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]

    elif which_probs == [1,13,23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_I + gap_13 + gap_23
        CI   = ( return_data["CI"],   1.-gap_I            )
        SI   = ( return_data["SI"],   1.-gap              )
        UIX  = ( return_data["UIX"],  1.-(gap_I + gap_23) )
        UIY  = ( return_data["UIY"],  1.-(gap_I + gap_13) )
        UIZ  = ( return_data["UIZ"],  1.-gap_I            )
        UIXY = ( return_data["UIXY"], 1.-gap              )
        UIXZ = ( return_data["UIXZ"], 1.-(gap_I + gap_23) )
        UIYZ = ( return_data["UIYZ"], 1.-(gap_I + gap_13) )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["CI"]   = sol_tx[1]
        return_data["SI"]   = sol_tx[2]
        return_data["UIX"]  = sol_tx[3]
        return_data["UIY"]  = sol_tx[4]
        return_data["UIZ"]  = sol_tx[5]
        return_data["UIXY"] = sol_tx[6]
        return_data["UIXZ"] = sol_tx[7]
        return_data["UIYZ"] = sol_tx[8]

    elif which_probs == [12,13,23]:
        print("MAXENT3D_PID.pid(): Recovering solution by Quadratic Programming")
        gap  = gap_12 + gap_13 + gap_23
        CI   = ( return_data["CI"],   0.                   )
        SI   = ( return_data["SI"],   1.-gap               )
        UIX  = ( return_data["UIX"],  1.-gap_23            )
        UIY  = ( return_data["UIY"],  1.-gap_13            )
        UIZ  = ( return_data["UIZ"],  1.-gap_12            )
        UIXY = ( return_data["UIXY"], 1.-(gap_13 + gap_23) )
        UIXZ = ( return_data["UIXZ"], 1.-(gap_12 + gap_23) )
        UIYZ = ( return_data["UIYZ"], 1.-(gap_12 + gap_13) )
        
        MI  = entropy_T  - condent__orig
        MIX = I_V(1,pdf)
        MIY = I_V(2,pdf)
        MIZ = I_V(3,pdf)
        
        recover_solver = QP(CI,SI,UIX,UIY,UIZ,UIXY,UIXZ,UIYZ,MI,MIX,MIY,MIZ)
        if output > 2: recover_solver.verbose = True
        c, G, h, dims, A, b = recover_solver.create_model(which_probs)
        recover_solver.ecos_kwargs = QP_solver_args
        retval, sol_tx, sol_slack, sol_lambda, sol_mu, sol_info = recover_solver.solve(c, G, h, dims, A, b, output)
        return_data["SI"]   = sol_tx[1]
        return_data["UIX"]  = sol_tx[2]
        return_data["UIY"]  = sol_tx[3]
        return_data["UIZ"]  = sol_tx[4]
        return_data["UIXY"] = sol_tx[5]
        return_data["UIXZ"] = sol_tx[6]
        return_data["UIYZ"] = sol_tx[7]
    #^ if which problem
    itoc_recover = time.process_time()
    if which_probs != [] and output > 0: print("\nMAXENT3D_PID.pid(): Time to recover solution is:", itoc_recover - itic_recover," secs") 
    # Sanity check

    #Check: MI(T; X,Y,Z) = SI + CI + UIX + UIY + UIZ + UIXY + UIXZ + UIYZ
    tol  = 1.e-10
    vio_T_XYZ = abs(entropy_T - condent__orig
               - return_data['CI'] - return_data['SI']
               - return_data['UIX'] - return_data['UIY'] - return_data['UIZ']
               - return_data['UIXY'] - return_data['UIXZ'] - return_data['UIYZ'])
    
    assert vio_T_XYZ < tol,                                 "MAXENT3D_PID.pid(): PID quantities must  sum up to mutual information, the violation is "+str(vio_T_XYZ)+", and the precision is set to "+str(tol)

    # Check: MI(T; X) = SI + UIX + UIXY + UIXZ
    vio_T_X = abs( I_V(1,pdf)
                - return_data['SI'] - return_data['UIX'] - return_data['UIXY'] - return_data['UIXZ'] )
    assert vio_T_X < tol, "MAXENT3D_PID.pid(): Unique and shared of X must sum up to MI(T; X), the violation is"+str(vio_T_X)+" and the precision is set to "+str(tol)

    # Check: MI(T; Y) = SI + UIY + UIXY + UIYZ
    vio_T_Y = abs( I_V(2,pdf)
                - return_data['SI'] - return_data['UIY'] - return_data['UIXY'] - return_data['UIYZ'] )
    
    assert vio_T_Y < tol, "MAXENT3D_PID.pid(): Unique and shared of Y must sum up to MI(T; Y), and the violation is"+str(vio_T_Y)+", and the precision is set to"+str(tol)


    # Check: MI(T; Z) = SI + UIZ + UIXZ + UIYZ
    vio_T_Z = abs( I_V(3,pdf)
                - return_data['SI'] - return_data['UIZ'] - return_data['UIXZ'] - return_data['UIYZ'])
    
    assert vio_T_Z < tol, "MAXENT3D_PID.pid(): Unique and shared of Z must sum up to MI(S; Z), the violation is"+str(vio_T_Z)+", and the precision is set to"+str(tol)

    return return_data
#^ pid()

#EOF
