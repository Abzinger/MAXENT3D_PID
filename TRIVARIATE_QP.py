"""TRIVARIATE_QP.py -- Python Module

Creates a Quadratic Program when one or more of the following

Optimization problems:

min - H(S|XYZ), min -H(S|XY), min -H(S|XZ) and min -H(S|YZ)

bails w/ duality gap > 10^-6  

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

                 x >= 0

                (W^(1/2), t, 1)in SOC

           In ECOS 

               min c^T[t,x]

               subject to

                     Ax = b

                     [-I, W^(1/2)]*[t,x]^T <_{K} [0, 0, 0]

           where K := R_+ . Q, c^T := [1/2,f^T]

        Args:
             which_probs: 
                         list(int) - [1]  if min_{Delta_p}H(T|X,Y,Z) failed

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

    if which_probs == [1] or which_probs == [1,12] or which_probs == [1,13] or which_probs ==[1,23] or which_probs == [1, 12, 13] or which_probs == [1, 12, 23] or which_probs == [1, 13, 23]:

        # Construct c 
        # c^T = [1/2, -CI_c*CI_e, -SI_c*SI_e,
        #        -UIX_c*UIX_e, -UIY_c*UIY_e, -UIZ_c*UIZ_e,
        #        -UIXY_c*UIXY_e, -UIXZ_c*UIXZ_e, -UIYZ_c*UIYZ_e]
        # x   = [t, CI, SI,
        #        UIX, UIY, UIZ,
        #        UIXY, UIXZ, UIYZ]
        self.c = np.array([.5, -self.CI_c*self.CI_e, -self.SI_c*self.SI_e,
                           -self.UIX_c*self.UIX_e, -self.UIY_c*self.UIY_e, -self.UIZ_c*self.UIZ_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIXZ_c*self.UIXZ_e, -self.UIYZ_c*self.UIYZ_e]) 
        
        # Construct A
        # A = [ [0,1,1,1,1,1,1,1,1],
        #       [0,0,1,1,0,0,1,1,0],
        #       [0,0,1,0,1,0,1,0,1],
        #       [0,0,1,0,0,1,0,1,1] ]
        
        Eqn = [0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1,
               2, 2, 2, 2,
               3, 3, 3, 3]
        Var = [1, 2, 3, 4, 5, 6, 7, 8,
               2, 3, 6, 7,
               2, 4, 6, 8,
               2, 5, 7, 8]
        Coeff = [+1., +1., +1., +1., +1., +1., +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1., +1., +1.]
        
        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,9), dtype=np.double)

        # Construct b 
        self.b = np.array([self.MI, self.MIX, self.MIY, self.MIZ])

        # Construct G
        # -G = [ [0,1,0,0,0,0,0,0,0], # CI >= 0
        #        [0,0,1,0,0,0,0,0,0], # SI >= 0
        #        [0,0,0,1,0,0,0,0,0], # UIX >= 0
        #        [0,0,0,0,1,0,0,0,0], # UIY >= 0
        #        [0,0,0,0,0,1,0,0,0], # UIZ >= 0
        #        [0,0,0,0,0,0,1,0,0], # UIXY >= 0
        #        [0,0,0,0,0,0,0,1,0], # UIXZ >= 0
        #        [0,0,0,0,0,0,0,0,1], # UIYZ >= 0
        #        [0,1,0,0,0,0,0,0,0], # CI >= CI_e
        #        [0,1,0,0,0,0,0,0,0], # SI <= SI_e
        #        [0,0,1,0,0,0,0,0,0], # UIX <= UIX_e 
        #        [0,0,0,1,0,0,0,0,0], # UIY <= UIY_e
        #        [0,0,0,0,1,0,0,0,0], # UIZ <= UIZ_e
        #        [0,0,0,0,0,1,0,0,0], # UIXY >= UIXY_e
        #        [0,0,0,0,0,0,1,0,0], # UIXZ >= UIXZ_e
        #        [0,0,0,0,0,0,0,1,0], # UIYZ >= UIYZ_e
        #        [1,0,0,0,0,0,0,0,0],
        #        [0,*,0,0,0,0,0,0,0],
        #        [0,0,*,0,0,0,0,0,0],
        #        [0,0,0,*,0,0,0,0,0],
        #        [0,0,0,0,*,0,0,0,0],
        #        [0,0,0,0,0,*,0,0,0],
        #        [0,0,0,0,0,0,*,0,0],
        #        [0,0,0,0,0,0,0,*,0],
        #        [0,0,0,0,0,0,0,0,*] ]
        
        Ieq   = [0,1,2,3,4,5,6,7,
                 8,9,10,11,12,13,14,15,
                 16,
                 17,18,
                 19,20,21,
                 22,23,24]
        Var   = [1,2,3,4,5,6,7,8,
                 1,2,3,4,5,6,7,8,
                 0,
                 1,2,
                 3,4,5,
                 6,7,8]
        Coeff = [-1., -1., -1., -1., -1., -1., -1., -1.,
                 -1., +1., +1., +1., +1., -1., -1., -1.,
                 -1.,
                 -self.CI_c, -self.SI_c,
                 -self.UIX_c, -self.UIY_c, -self.UIZ_c,
                 -self.UIXY_c, -self.UIXZ_c, -self.UIYZ_c] # confidence values 1/dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(25,9), dtype=np.double)

        # Construct K 
        self.dims['l'] = 16
        self.dims['q'] = [9]

        # Construct h
        self.h = np.zeros( (25,),dtype=np.double )
        self.h[8] = -self.CI_e
        self.h[9] = self.SI_e

        self.h[10] = self.UIX_e
        self.h[11] = self.UIY_e
        self.h[12] = self.UIZ_e

        self.h[13] = -self.UIXY_e
        self.h[14] = -self.UIXZ_e
        self.h[15] = -self.UIYZ_e
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XYZ)]
 
    if which_probs == [12]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIZ_c*UIZ_e,
        #        -UIXZ_c*UIXZ_e, -UIYZ_c*UIYZ_e]
        # x   = [t, SI,
        #           UIZ,
        #           UIXZ, UIYZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIZ_c*self.UIZ_e,
                           -self.UIXZ_c*self.UIXZ_e, -self.UIYZ_c*self.UIYZ_e]) 
        # Construct A
        # A = [ [0,1,1,1,1],
        #       [0,1,0,1,0],
        #       [0,1,0,0,1],
        #       [0,1,1,1,1] ]
        Eqn = [0, 0, 0, 0,
               1, 1,
               2, 2,
               3, 3, 3, 3]
        Var = [1, 2, 3, 4,
               1, 3,
               1, 4,
               1, 2, 3, 4]
        Coeff = [+1., +1., +1., +1.,
                 +1., +1.,
                 +1., +1.,
                 +1., +1., +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,5), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e - self.UIX_e - self.UIY_e - self.UIXY_e,
                           self.MIX - self.UIX_e - self.UIXY_e,
                           self.MIY - self.UIY_e - self.UIXY_e,
                           self.MIZ])

        # Construct G
        # -G = [ [0,1,0,0,0],
        #        [0,0,1,0,0],
        #        [0,0,0,1,0],
        #        [0,0,0,0,1],
        #        [1,0,0,0,0],
        #        [0,*,0,0,0],
        #        [0,0,*,0,0],
        #        [0,0,0,*,0],
        #        [0,0,0,0,*], ]
        
        Ieq   = [0,1,2,3,
                 4,
                 5,
                 6,
                 7,8]
        Var   = [1,2,3,4,
                 0,
                 1,
                 2,
                 3,4]
        Coeff = [-1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIZ_c,
                 -self.UIXZ_c, -self.UIYZ_c] # confidence values 1/dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(9,5), dtype=np.double)

        # Construct K 
        self.dims['l'] = 4
        self.dims['q'] = [5]

        # Construct h
        self.h = np.zeros( (9,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XY)]

    if which_probs == [13]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIY_c*UIY_e,
        #        -UIXY_c*UIXY_e, -UIYZ_c*UIYZ_e]
        # x   = [t, SI,
        #        UIY,
        #        UIXY, UIYZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIY_c*self.UIY_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIYZ_c*self.UIYZ_e]) 

        # Construct A
        # A = [ [0,1,1,1,1],
        #       [0,1,0,1,0],
        #       [0,1,1,1,1],
        #       [0,1,0,0,1] ]
        Eqn = [0, 0, 0, 0,
               1, 1,
               2, 2, 2, 2,
               3, 3]
        Var = [1, 2, 3, 4,
               1, 3,
               1, 2, 3, 4,
               1, 4]
        Coeff = [+1., +1., +1., +1.,
                 +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,5), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e - self.UIX_e - self.UIZ_e - self.UIXZ_e,
                           self.MIX - self.UIX_e - self.UIXZ_e,
                           self.MIY,
                           self.MIZ - self.UIZ_e - self.UIXZ_e])

        # Construct G
        # -G = [ [0,1,0,0,0],
        #        [0,0,1,0,0],
        #        [0,0,0,1,0],
        #        [0,0,0,0,1],
        #        [1,0,0,0,0],
        #        [0,*,0,0,0],
        #        [0,0,*,0,0],
        #        [0,0,0,*,0],
        #        [0,0,0,0,*], ]
        Ieq   = [0,1,2,3,
                 4,
                 5,
                 6,
                 7,8]
        Var   = [1,2,3,4,
                 0,
                 1,
                 2,
                 3,4]
        Coeff = [-1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIY_c,
                 -self.UIXY_c, -self.UIYZ_c] # confidence values 1/dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(9,5), dtype=np.double)

        # Construct K 
        self.dims['l'] = 4
        self.dims['q'] = [5]

        # Construct h
        self.h = np.zeros( (9,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XZ)]

    if which_probs == [23]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIX_c*UIX_e,
        #        -UIXY_c*UIXY_e, -UIXZ_c*UIXZ_e]
        # x   = [t, SI,
        #        UIX,
        #        UIXY, UIXZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIX_c*self.UIX_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIXZ_c*self.UIXZ_e]) 

        # Construct A
        # A = [ [0,1,1,1,1],
        #       [0,1,1,1,1],
        #       [0,1,0,1,0],
        #       [0,1,0,0,1] ]
        Eqn = [0, 0, 0, 0,
               1, 1, 1, 1,
               2, 2,
               3, 3]
        Var = [1, 2, 3, 4,
               1, 2, 3, 4,
               1, 3,
               1, 4]
        Coeff = [+1., +1., +1., +1.,
                 +1., +1., +1., +1., 
                 +1., +1.,
                 +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,5), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e - self.UIY_e - self.UIZ_e - self.UIYZ_e,
                           self.MIX,
                           self.MIY - self.UIY_e - self.UIYZ_e,
                           self.MIZ - self.UIZ_e - self.UIYZ_e])

        # Construct G
        # -G = [ [0,1,0,0,0],
        #        [0,0,1,0,0],
        #        [0,0,0,1,0],
        #        [0,0,0,0,1],
        #        [1,0,0,0,0],
        #        [0,*,0,0,0],
        #        [0,0,*,0,0],
        #        [0,0,0,*,0],
        #        [0,0,0,0,*], ]
        Ieq   = [0,1,2,3,
                 4,
                 5,
                 6,
                 7,8]
        Var   = [1,2,3,4,
                 0,
                 1,
                 2,
                 3,4]
        Coeff = [-1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIX_c,
                 -self.UIXY_c, -self.UIXZ_c] # confidence values 1/dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(9,5), dtype=np.double)

        # Construct K 
        self.dims['l'] = 4
        self.dims['q'] = [5]

        # Construct h
        self.h = np.zeros( (9,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|YZ)]

    if which_probs == [12,13]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIY_c*UIY_e, -UIZ_c*UIZ_e,
        #        -UIXY_c*UIXY_e, -UIXZ_c*UIXZ_e, -UIYZ_c*UIYZ_e]
        # x   = [t, SI,
        #        UIY, UIZ,
        #        UIXY, UIXZ, UIYZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIY_c*self.UIY_e, -self.UIZ_c*self.UIZ_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIXZ_c*self.UIXZ_e, -self.UIYZ_c*self.UIYZ_e]) 

        # Construct A
        # A = [ [0,1,1,1,1,1,1],
        #       [0,1,0,0,1,1,0],
        #       [0,1,1,0,1,0,1],
        #       [0,1,0,1,0,1,1] ]
        Eqn = [0, 0, 0, 0, 0, 0,
               1, 1, 1,
               2, 2, 2, 2,
               3, 3, 3, 3]
        Var = [1, 2, 3, 4, 5, 6,
               1, 4, 5,
               1, 2, 4, 6,
               1, 3, 5, 6]
        Coeff = [+1., +1., +1., +1., +1., +1., 
                 +1., +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1., +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,7), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e - self.UIX_e,
                           self.MIX - self.UIX_e,
                           self.MIY,
                           self.MIZ])

        # Construct G
        # -G = [ [0,1,0,0,0,0,0],
        #        [0,0,1,0,0,0,0],
        #        [0,0,0,1,0,0,0],
        #        [0,0,0,0,1,0,0],
        #        [0,0,0,0,0,1,0],
        #        [0,0,0,0,0,0,1],
        #        [1,0,0,0,0,0,0],
        #        [0,*,0,0,0,0,0],
        #        [0,0,*,0,0,0,0],
        #        [0,0,0,*,0,0,0],
        #        [0,0,0,0,*,0,0],
        #        [0,0,0,0,0,*,0],
        #        [0,0,0,0,0,0,*], ]
        Ieq   = [0,1,2,3,4,5,
                 6,
                 7,
                 8,9,
                 10,11,12]
        Var   = [1,2,3,4,5,6,
                 0,
                 1,
                 2,3,
                 4,5,6]
        Coeff = [-1., -1., -1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIY_c, -self.UIZ_c,
                 -self.UIXY_c, -self.UIXZ_c, -self.UIYZ_c] # confidence values 1/2dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(13,7), dtype=np.double)

        # Construct K 
        self.dims['l'] = 6
        self.dims['q'] = [7]

        # Construct h
        self.h = np.zeros( (13,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XY)] and [min - H(S|XZ)]

    if which_probs == [12,23]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIX_c*UIX_e, -UIZ_c*UIZ_e,
        #        -UIXY_c*UIXY_e, -UIXZ_c*UIXZ_e, -UIYZ_c*UIYZ_e]
        # x   = [t, SI,
        #        UIX, UIZ,
        #        UIXY, UIXZ, UIYZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIX_c*self.UIX_e, -self.UIZ_c*self.UIZ_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIXZ_c*self.UIXZ_e, -self.UIYZ_c*self.UIYZ_e]) 

        # Construct A
        # A = [ [0,1,1,1,1,1,1],
        #       [0,1,1,0,1,1,0],
        #       [0,1,0,0,1,0,1],
        #       [0,1,0,1,0,1,1] ]
        Eqn = [0, 0, 0, 0, 0, 0,
               1, 1, 1, 1,
               2, 2, 2,
               3, 3, 3, 3]
        Var = [1, 2, 3, 4, 5, 6,
               1, 2, 4, 5,
               1, 4, 6,
               1, 3, 5, 6]
        Coeff = [+1., +1., +1., +1., +1., +1., 
                 +1., +1., +1., +1.,
                 +1., +1., +1.,
                 +1., +1., +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,7), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e - self.UIY_e,
                           self.MIX,
                           self.MIY - self.UIY_e,
                           self.MIZ])

        # Construct G
        # -G = [ [0,1,0,0,0,0,0],
        #        [0,0,1,0,0,0,0],
        #        [0,0,0,1,0,0,0],
        #        [0,0,0,0,1,0,0],
        #        [0,0,0,0,0,1,0],
        #        [0,0,0,0,0,0,1],
        #        [1,0,0,0,0,0,0],
        #        [0,*,0,0,0,0,0],
        #        [0,0,*,0,0,0,0],
        #        [0,0,0,*,0,0,0],
        #        [0,0,0,0,*,0,0],
        #        [0,0,0,0,0,*,0],
        #        [0,0,0,0,0,0,*], ]
        Ieq   = [0,1,2,3,4,5,
                 6,
                 7,
                 8,9,
                 10,11,12]
        Var   = [1,2,3,4,5,6,
                 0,
                 1,
                 2,3,
                 4,5,6]
        Coeff = [-1., -1., -1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIX_c, -self.UIZ_c,
                 -self.UIXY_c, -self.UIXZ_c, -self.UIYZ_c] # confidence values 1/2dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(13,7), dtype=np.double)

        # Construct K 
        self.dims['l'] = 6
        self.dims['q'] = [7]

        # Construct h
        self.h = np.zeros( (13,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XY)] and [min - H(S|YZ)]

    if which_probs == [13,23]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIX_c*UIX_e, -UIY_c*UIY_e,
        #        -UIXY_c*UIXY_e, -UIXZ_c*UIXZ_e, -UIYZ_c*UIYZ_e]
        # x   = [t, SI,
        #        UIX, UIY,
        #        UIXY, UIXZ, UIYZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIX_c*self.UIX_e, -self.UIY_c*self.UIY_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIXZ_c*self.UIXZ_e, -self.UIYZ_c*self.UIYZ_e])
        
        # Construct A
        # A = [ [0,1,1,1,1,1,1],
        #       [0,1,1,0,1,1,0],
        #       [0,1,0,1,1,0,1],
        #       [0,1,0,0,0,1,1] ]
        Eqn = [0, 0, 0, 0, 0, 0,
               1, 1, 1, 1,
               2, 2, 2, 2,
               3, 3, 3]
        Var = [1, 2, 3, 4, 5, 6,
               1, 2, 4, 5,
               1, 3, 4, 6,
               1, 5, 6]
        Coeff = [+1., +1., +1., +1., +1., +1., 
                 +1., +1., +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,7), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e - self.UIZ_e,
                           self.MIX,
                           self.MIY,
                           self.MIZ - self.UIZ_e])

        # Construct G
        # -G = [ [0,1,0,0,0,0,0],
        #        [0,0,1,0,0,0,0],
        #        [0,0,0,1,0,0,0],
        #        [0,0,0,0,1,0,0],
        #        [0,0,0,0,0,1,0],
        #        [0,0,0,0,0,0,1],
        #        [1,0,0,0,0,0,0],
        #        [0,*,0,0,0,0,0],
        #        [0,0,*,0,0,0,0],
        #        [0,0,0,*,0,0,0],
        #        [0,0,0,0,*,0,0],
        #        [0,0,0,0,0,*,0],
        #        [0,0,0,0,0,0,*], ]
        Ieq   = [0,1,2,3,4,5,
                 6,
                 7,
                 8,9,
                 10,11,12]
        Var   = [1,2,3,4,5,6,
                 0,
                 1,
                 2,3,
                 4,5,6]
        Coeff = [-1., -1., -1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIX_c, -self.UIY_c,
                 -self.UIXY_c, -self.UIXZ_c, -self.UIYZ_c] # confidence values 1/2dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(13,7), dtype=np.double)

        # Construct K 
        self.dims['l'] = 6
        self.dims['q'] = [7]

        # Construct h
        self.h = np.zeros( (13,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XZ)] and [min - H(S|YZ)]

    if which_probs == [12,13,23]:
        # Construct c 
        # c^T = [1/2, -SI_c*SI_e,
        #        -UIX_c*UIX_e, -UIY_c*UIY_e, -UIZ_c*UIZ_e,
        #        -UIXY_c*UIXY_e, -UIXZ_c*UIXZ_e, -UIYZ_c*UIYZ_e]
        # x   = [t, SI,
        #        UIX, UIY, UIZ,
        #        UIXY, UIXZ, UIYZ]
        self.c = np.array([.5, -self.SI_c*self.SI_e,
                           -self.UIX_c*self.UIX_e, -self.UIY_c*self.UIY_e, -self.UIZ_c*self.UIZ_e,
                           -self.UIXY_c*self.UIXY_e, -self.UIXZ_c*self.UIXZ_e, -self.UIYZ_c*self.UIYZ_e]) 

        # Construct A
        # A = [ [0,1,1,1,1,1,1,1],
        #       [0,1,1,0,0,1,1,0],
        #       [0,1,0,1,0,1,0,1],
        #       [0,1,0,0,1,0,1,1] ]
        Eqn = [0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1,
               2, 2, 2, 2,
               3, 3, 3, 3]
        Var = [1, 2, 3, 4, 5, 6, 7,
               1, 2, 5, 6,
               1, 3, 5, 7,
               1, 4, 6, 7]
        Coeff = [+1., +1., +1., +1., +1., +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1., +1., +1.,
                 +1., +1., +1., +1.]

        self.A = sparse.csc_matrix( (Coeff, (Eqn,Var)), shape=(4,8), dtype=np.double)

        # Construct b
        self.b = np.array([self.MI - self.CI_e,
                           self.MIX,
                           self.MIY,
                           self.MIZ])

        # Construct G
        # -G = [ [0,1,0,0,0,0,0,0],
        #        [0,0,1,0,0,0,0,0],
        #        [0,0,0,1,0,0,0,0],
        #        [0,0,0,0,1,0,0,0],
        #        [0,0,0,0,0,1,0,0],
        #        [0,0,0,0,0,0,1,0],
        #        [0,0,0,0,0,0,0,1],
        #        [1,0,0,0,0,0,0,0],
        #        [0,*,0,0,0,0,0,0],
        #        [0,0,*,0,0,0,0,0],
        #        [0,0,0,*,0,0,0,0],
        #        [0,0,0,0,*,0,0,0],
        #        [0,0,0,0,0,*,0,0],
        #        [0,0,0,0,0,0,*,0],
        #        [0,0,0,0,0,0,0,*], ]

        # Construct W^(1\2) 
        # W^(1/2) = w^(1\2)I of dim = 8 where w is the confidence of the nonconvergent problem        
        Ieq   = [0,1,2,3,4,5,6,
                 7,
                 8,
                 9,10,11,
                 12,13,14]
        Var   = [1,2,3,4,5,6,7,
                 0,
                 1,
                 2,3,4,
                 5,6,7]
        Coeff = [-1., -1., -1., -1., -1., -1., -1.,
                 -1.,
                 -self.SI_c,
                 -self.UIX_c, -self.UIY_c, -self.UIZ_c,
                 -self.UIXY_c, -self.UIXZ_c, -self.UIYZ_c] # confidence values 1/2dual_gap
        self.G = sparse.csc_matrix( (Coeff, (Ieq,Var)), shape=(15,8), dtype=np.double)

        # Construct K 
        self.dims['l'] = 7
        self.dims['q'] = [8]

        # Construct h
        self.h = np.zeros( (15,),dtype=np.double )
        
        return self.c, self.G, self.h, self.dims, self.A, self.b
    #^ if [min - H(S|XY)], [min - H(S|XZ)], and [min - H(S|YZ)]

    
#^ create_model()

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
    itic = time.process_time()
    
    if self.verbose != None:
        # print(self.verbose)
        self.ecos_kwargs["verbose"] = self.verbose
    #^ if
    
    solution = ecos.solve(c, G, h, dims, A, b, **self.ecos_kwargs)

    if 'x' in solution.keys():
        self.sol_tx     = solution['x']
        self.sol_slack  = solution['s']
        self.sol_lambda = solution['y']
        self.sol_mu     = solution['z']
        self.sol_info   = solution['info']
        itoc = time.process_time()
        if output == 2: print("TRIVARIATE_QP.solve(): Time to solve the Quadratic Program of Least Squares", itoc - itic, "secs") 
        return "success", self.sol_tx, self.sol_slack, self.sol_lambda, self.sol_mu, self.sol_info
    else: # "x" not in dict solution
        return "TRIVARIATE_QP.solve(): x not in dict solution -- No Solution Found!!!"
    #^ if/esle
#^ solve()
