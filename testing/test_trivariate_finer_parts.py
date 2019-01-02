"""test_chicharro_subparts.py
a python script to test the hierarchy of finding the finer Trivariate PID
"""
import numpy as np
from numpy.linalg import inv
from collections import defaultdict
import math 
import BROJA_2PID
import Chicharro_PID
from Chicharro_PID import I_VW, I_XYZ, I_V

log = math.log2


# AND DUBLICATE
andDgate = dict()
andDgate[ (0,0,0,0) ] = .25
andDgate[ (0,0,1,0) ] = .25
andDgate[ (0,1,0,1) ] = .25
andDgate[ (1,1,1,1) ] = .25

# XOR gate
xorgate = dict()
xorgate[ (0,0,0,0) ] = .125
xorgate[ (1,0,0,1) ] = .125
xorgate[ (1,0,1,0) ] = .125
xorgate[ (1,1,0,0) ] = .125
xorgate[ (0,0,1,1) ] = .125
xorgate[ (0,1,0,1) ] = .125
xorgate[ (0,1,1,0) ] = .125
xorgate[ (1,1,1,1) ] = .125

# XOR Multi Core
# a -> 0 ; b -> 2 ; c -> 4; A -> 1; B -> 3; C -> 5
xorMgate = dict()
xorMgate[ ( 0, (0,2),(0,4),(2,4) ) ] = .125
xorMgate[ ( 0, (1,3),(1,4),(3,4) ) ] = .125
xorMgate[ ( 0, (1,2),(1,5),(2,5) ) ] = .125
xorMgate[ ( 0, (0,3),(0,5),(3,5) ) ] = .125
xorMgate[ ( 1, (1,2),(1,4),(2,4) ) ] = .125
# xorMgate[ ( 1, (0,3),(0,4),(2,4) ) ] = .125
xorMgate[ ( 1, (0,3),(0,4),(3,4) ) ] = .125
# xorMgate[ ( 1, (0,2),(0,4),(2,5) ) ] = .125
xorMgate[ ( 1, (0,2),(0,5),(2,5) ) ] = .125
xorMgate[ ( 1, (1,3),(1,5),(3,5) ) ] = .125

# XOR Loss
xorLgate = dict()
xorLgate[ (0,0,0,0) ] = .25
xorLgate[ (1,0,1,1) ] = .25
xorLgate[ (1,1,0,1) ] = .25
xorLgate[ (0,1,1,0) ] = .25

# XOR Duplicate
xorDgate = dict()
xorDgate[ (0,0,0,0) ] = 0.25
xorDgate[ (1,0,1,0) ] = 0.25
xorDgate[ (1,1,0,1) ] = 0.25
xorDgate[ (0,1,1,1) ] = 0.25

# XY XOR gate
xorXYgate = dict()
xorXYgate[ (0,0,0,2) ] = 0.25
xorXYgate[ (1,1,0,2) ] = 0.25
xorXYgate[ (1,0,1,2) ] = 0.25
xorXYgate[ (0,1,1,2) ] = 0.25

# X UNQ gate
unqXgate = dict()
unqXgate[ (0,0,0,2) ] = 0.25
unqXgate[ (0,0,1,2) ] = 0.25
unqXgate[ (1,1,0,2) ] = 0.25
unqXgate[ (1,1,1,2) ] = 0.25

# X Z SUNQ gate
sunqXgate = dict()
sunqXgate[ (0,0,0,0) ] = 0.25
sunqXgate[ (0,0,1,0) ] = 0.25
sunqXgate[ (1,1,0,1) ] = 0.25
sunqXgate[ (0,1,1,1) ] = 0.25

# XORAND gate: (X xor Y) and Z 
xorandgate = dict()
xorandgate[ (0,0,0,0) ] = 0.125
xorandgate[ (0,0,1,0) ] = 0.125
xorandgate[ (0,1,0,0) ] = 0.125
xorandgate[ (0,1,1,0) ] = 0.125
xorandgate[ (0,0,0,1) ] = 0.125
xorandgate[ (1,0,1,1) ] = 0.125
xorandgate[ (1,1,0,1) ] = 0.125
xorandgate[ (0,1,1,1) ] = 0.125

# XORCOPY gate: T = ( (X xor Y), Z )
xorcopygate = dict()
xorcopygate[ ((0,0),0,0,0) ] = 0.125
xorcopygate[ ((1,0),0,1,0) ] = 0.125
xorcopygate[ ((1,0),1,0,0) ] = 0.125
xorcopygate[ ((0,0),1,1,0) ] = 0.125
xorcopygate[ ((0,1),0,0,1) ] = 0.125
xorcopygate[ ((1,1),0,1,1) ] = 0.125
xorcopygate[ ((1,1),1,0,1) ] = 0.125
xorcopygate[ ((0,1),1,1,1) ] = 0.125

# positive gate
pgate = dict()
pgate[ (0,0,0,0) ] = .09375 
pgate[ (1,0,1,1) ] = .09375 
pgate[ (1,1,0,0) ] = .09375 
pgate[ (0,1,1,0) ] = .09375 

pgate[ (1,0,0,0) ] = .03125
pgate[ (0,0,1,1) ] = .03125
pgate[ (0,1,0,0) ] = .03125
pgate[ (1,1,1,0) ] = .03125

pgate[ (0,0,0,1) ] = .0625 
pgate[ (0,1,0,1) ] = .0625 
pgate[ (0,0,1,0) ] = .0625 
pgate[ (0,1,1,1) ] = .0625 

pgate[ (1,0,0,1) ] = .0625 
pgate[ (1,1,0,1) ] = .0625 
pgate[ (1,0,1,0) ] = .0625 
pgate[ (1,1,1,1) ] = .0625 

def create_subsystems(pdf):
  """Creates the Subsystems needed for the hierarchy to 
     compute the finer parts of Trivariate PID

     Args:
         pdf: dict - the original distribution P of (T,X,Y,Z)
              keys: (t,x,y,z)
              values: P(t,x,y,z)

     Return: subsystems: dict - a dictionary of dictionaries containing
             the need subsystems of (T,X,Y,Z)
             keys: string - ex: '(T,(X,Y),Z)'
             values: dict - pdf of (T,(X,Y),Z)
  """

  subsystems = defaultdict(dict)

  # txysys = defaultdict(lambda :0.)
  # txzsys = defaultdict(lambda :0.)
  # tyzsys = defaultdict(lambda :0.)

  txy_zsys = defaultdict(lambda :0.)
  txz_ysys = defaultdict(lambda :0.)
  tyz_xsys = defaultdict(lambda :0.)

  txy_xzsys = defaultdict(lambda :0.)
  txy_yzsys = defaultdict(lambda :0.)
  txz_yzsys = defaultdict(lambda :0.)

  txy_xz_yzsys = defaultdict(lambda :0.)

  for txyz,v in pdf.items():
    t,x,y,z = txyz
    
    txy_zsys[ (t,(x,y),z) ] += v
    txz_ysys[ (t,(x,z),y) ] += v
    tyz_xsys[ (t,(y,z),x) ] += v

    txy_xzsys[ (t,(x,y),(x,z)) ] += v
    txy_yzsys[ (t,(x,y),(y,z)) ] += v
    txz_yzsys[ (t,(x,z),(y,z)) ] += v

    txy_xz_yzsys[ (t,(x,y),(x,z),(y,z)) ] += v
  #^ for
  
  subsystems['(T,(X,Y),Z)'] = dict(txy_zsys)
  subsystems['(T,(X,Z),Y)'] = dict(txz_ysys)
  subsystems['(T,(Y,Z),X)'] = dict(tyz_xsys)

  subsystems['(T,(X,Y),(X,Z))'] = dict(txy_xzsys)
  subsystems['(T,(X,Y),(Y,Z))'] = dict(txy_yzsys)
  subsystems['(T,(X,Z),(Y,Z))'] = dict(txz_yzsys)

  subsystems['(T,(X,Y),(X,Z),(Y,Z))'] = dict(txy_xz_yzsys)
  subsystems['(T,X,Y,Z)'] = pdf
  return subsystems
#^ create subsystems


def compute_pid(gate):
  """Compute the PID of (T,X,Y,Z) and its subsystems given by
     create_subsystems()
  
     Args:
         gate: dict - the original distribution P of (T,X,Y,Z)
               keys: (t,x,y,z)
               values: P(t,x,y,z)
  
     Returns:
         uniques: dict - contains the values of unique,
                  redundant unique, and redundant information
                  keys: string - ex: 'UI(T;Z\X,Y)'
                  values: float - ex: UI(T;Z\X,Y)
       
         synergys: numpy.array(8) - contains the values of 
                   R.H.S. of the system of equations of synergys
  """
  subsystems = create_subsystems(gate)
  synergys = np.zeros(8, dtype=float)
  uniques = dict()
  counter = 1
  SIXYZ = 0.
  SIYXZ = 0.
  SIZXY = 0.
  for system,pdf in subsystems.items():
    if system != '(T,X,Y,Z)' and system != '(T,(X,Y),(X,Z),(Y,Z))':
      # ECOS parameters 
      parms = dict()
      parms['max_iters'] = 100
      parms['keep_solver_object'] = True

      # Start computing 
      print("Starting BROJA_2PID.pid() on "+system+" subsystem.")
      returndict = BROJA_2PID.pid(pdf, cone_solver="ECOS", output=0, **parms)

      msg="""PID of """+system+"""
      Shared information: {SI}
      Unique information in X1: {UIY}
      Unique information in X2: {UIZ}
      Synergistic information: {CI}
      Primal feasibility: {Num_err[0]}
      Dual feasibility: {Num_err[1]}
      Duality Gap: {Num_err[2]}"""
      # print(msg.format(**returndict))
      synergys[counter] = returndict['CI']
      counter += 1
      if system == '(T,(X,Y),Z)':
        uniques['UI(T;Z\X,Y)'] = returndict['UIZ']
        SIZXY = returndict['SI']
      if system == '(T,(X,Z),Y)':
        uniques['UI(T;Y\X,Z)'] = returndict['UIZ']
        SIYXZ= returndict['SI']
      if system == '(T,(Y,Z),X)':
        uniques['UI(T;X\Y,Z)'] = returndict['UIZ']
        SIXYZ = returndict['SI']
    elif system == '(T,X,Y,Z)':
      # ECOS parameters 
      parms = dict()
      parms['max_iters'] = 100
      parms['keep_solver_object'] = True

      # Start Computing
      print("Starting Chicharro_PID.pid() on "+system+" subsystem.")

      returndict = Chicharro_PID.pid(pdf, cone_solver="ECOS", parallel ='on', output=0, **parms)
      msg="""PID of"""+system+"""
      Synergistic information: {CI}
      Unique information in X: {UIX}
      Unique information in Y: {UIY}
      Unique information in Z: {UIZ}
      Unique information in X,Y: {UIXY}
      Unique information in X,Z: {UIXZ}
      Unique information in Y,Z: {UIYZ}
      Shared information: {SI}
      Primal feasibility ( min MI(S;X,Y,Z) ): {Num_err_I[0]}
      Dual feasibility ( min MI(S;X,Y,Z) ): {Num_err_I[1]}
      Duality Gap ( min MI(S;X,Y,Z) ): {Num_err_I[2]}
      Primal feasibility ( min MI(S;X,Y) ): {Num_err_12[0]}
      Dual feasibility ( min MI(S;X,Y) ): {Num_err_12[1]}
      Duality Gap ( min MI(S;X,Y) ): {Num_err_12[2]}
      Primal feasibility ( min MI(S;X,Z) ): {Num_err_13[0]}
      Dual feasibility ( min MI(S|;,Z) ): {Num_err_13[1]}
      Duality Gap ( min MI(S;X,Z) ): {Num_err_13[2]}
      Primal feasibility ( min MI(S;Y,Z) ): {Num_err_23[0]}
      Dual feasibility ( min MI(S;Y,Z) ): {Num_err_23[1]}
      Duality Gap ( min MI(S;Y,Z) ): {Num_err_23[2]}"""
      # print(msg.format(**returndict))
      chi_pid = returndict
      synergys[0] = returndict['CI']

      uniques['SI(T;Z,X:Y)'] = SIZXY - returndict['SI'] - returndict['UIXZ'] - returndict['UIYZ']
      uniques['SI(T;Y,X:Z)'] = SIYXZ - returndict['SI'] - returndict['UIXY'] - returndict['UIYZ']
      uniques['SI(T;X,Y:Z)'] = SIXYZ - returndict['SI'] - returndict['UIXY'] - returndict['UIXZ']

      uniques['UI(T;X,Y\Z)'] = returndict['UIXY']
      uniques['UI(T;X,Z\Y)'] = returndict['UIXZ']
      uniques['UI(T;Y,Z\X)'] = returndict['UIYZ']
      
      uniques['SI(T;X,Y,Z)']  = returndict['SI']
    else:
      # ECOS parameters 
      parms = dict()
      parms['max_iters'] = 100
      parms['keep_solver_object'] = True

      # Start computing
      print("Starting Chicharro_PID.pid() on "+system+" subsystem.")

      returndict = Chicharro_PID.pid(pdf, cone_solver="ECOS", parallel ='on', output=0, **parms)
      msg="""PID of """+system+"""
      Synergistic information: {CI}
      Unique information in X: {UIX}
      Unique information in Y: {UIY}
      Unique information in Z: {UIZ}
      Unique information in X,Y: {UIXY}
      Unique information in X,Z: {UIXZ}
      Unique information in Y,Z: {UIYZ}
      Shared information: {SI}
      Primal feasibility ( min MI(S;X,Y,Z) ): {Num_err_I[0]}
      Dual feasibility ( min MI(S;X,Y,Z) ): {Num_err_I[1]}
      Duality Gap ( min MI(S;X,Y,Z) ): {Num_err_I[2]}
      Primal feasibility ( min MI(S;X,Y) ): {Num_err_12[0]}
      Dual feasibility ( min MI(S;X,Y) ): {Num_err_12[1]}
      Duality Gap ( min MI(S;X,Y) ): {Num_err_12[2]}
      Primal feasibility ( min MI(S;X,Z) ): {Num_err_13[0]}
      Dual feasibility ( min MI(S|;,Z) ): {Num_err_13[1]}
      Duality Gap ( min MI(S;X,Z) ): {Num_err_13[2]}
      Primal feasibility ( min MI(S;Y,Z) ): {Num_err_23[0]}
      Dual feasibility ( min MI(S;Y,Z) ): {Num_err_23[1]}
      Duality Gap ( min MI(S;Y,Z) ): {Num_err_23[2]}"""
      # print(msg.format(**returndict))
      synergys[7] = returndict['CI']
      #^ if
  #^ for
  return uniques, synergys
#^ compute_pid()


def find_subparts(uniques, synergys):
  """Computes the finer parts of PID of (T,X,Y,Z)
  
     Args:
         uniques: dict - contains the values of unique,
                  redundant unique, and redundant information
                  keys: string - ex: 'UI(T;Z\X,Y)'
                  values: float - ex: UI(T;Z\X,Y)
       
         synergys: numpy.array(8) - contains the values of 
                   R.H.S. of the system of equations of synergys

    Return:
         tripid: dict - contains the values of inividual synergy,
                 unique, redundant unique, and redundant information
                  keys: string - ex: 'UI(T;Z\X,Y)'
                  values: float - ex: UI(T;Z\X,Y)
  """
  
  A = np.array([
  [1.,1.,1.,1.,1.,1.,1.,1.], # syn tx_y_z 
    
  [1.,0.,1.,1.,0.,0.,1.,0.], # syn txy_z
  [1.,1.,0.,1.,0.,1.,0.,0.], # syn txz_y
  [1.,1.,1.,0.,1.,0.,0.,0.], # syn tyz_x
    
  [1.,0.,0.,1.,0.,0.,0.,0.], # syn txy_xz
  [1.,0.,1.,0.,0.,0.,0.,0.], # syn txy_yz
  [1.,1.,0.,0.,0.,0.,0.,0.], # syn txz_yz
    
  [1.,0.,0.,0.,0.,0.,0.,0.], # syn txy_xz_yz
  ])
  sol = np.linalg.solve(A, synergys)
  tripid = dict()
  tripid['CI(T;X;Y;Z)']       = sol[0]
  tripid['CI(T;X;Y)']         = sol[1]
  tripid['CI(T;X;Z)']         = sol[2]
  tripid['CI(T;Y;Z)']         = sol[3]
  tripid['CI(T;X;Y,X;Z)']     = sol[4]
  tripid['CI(T;X;Y,Y;Z)']     = sol[5]
  tripid['CI(T;X;Z,Y;Z)']     = sol[6]
  tripid['CI(T;X;Y,X;Z,Y;Z)'] = sol[7]

  tripid['UI(T;Z\X,Y)'] = uniques['UI(T;Z\X,Y)']
  tripid['UI(T;Y\X,Z)'] = uniques['UI(T;Y\X,Z)']
  tripid['UI(T;X\Y,Z)'] = uniques['UI(T;X\Y,Z)']
  
  tripid['SI(T;Z,X;Y)'] = uniques['SI(T;Z,X:Y)']
  tripid['SI(T;Y,X;Z)'] = uniques['SI(T;Y,X:Z)']
  tripid['SI(T;X,Y;Z)'] = uniques['SI(T;X,Y:Z)']

  tripid['UI(T;X,Y\Z)'] = uniques['UI(T;X,Y\Z)']
  tripid['UI(T;X,Z\Y)'] = uniques['UI(T;X,Z\Y)'] 
  tripid['UI(T;Y,Z\X)'] = uniques['UI(T;Y,Z\X)']

  tripid['SI(T;X,Y,Z)'] = uniques['SI(T;X,Y,Z)']

  return tripid
#^ find_subparts()

def main():
  """ Find the PID of different given gates
  """
  
  print("AND Duplicate gate: ")
  uniques, synergys = compute_pid(andDgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("XOR gate: ")
  uniques, synergys = compute_pid(xorgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("XOR Duplicate gate: ")
  uniques, synergys = compute_pid(xorDgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("XOR Multi core gate: ")
  uniques, synergys = compute_pid(xorMgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("XOR Loss gate: ")
  uniques, synergys = compute_pid(xorLgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("XOR XY gate: ")
  uniques, synergys = compute_pid(xorXYgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("UNQ X gate: ")
  uniques, synergys = compute_pid(unqXgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")

  print("SUNQ X gate: ")
  uniques, synergys = compute_pid(sunqXgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")
  
  print("XORAND gate: ")
  uniques, synergys = compute_pid(xorandgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")
  
  print("XORCOPY gate: ")
  uniques, synergys = compute_pid(xorcopygate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))
  print("\n...\n")
  
  print("positive gate: ")
  uniques, synergys = compute_pid(pgate)
  tripid = find_subparts(uniques, synergys)
  msg="""PID of (T,X,Y,Z)
  CI(T;X:Y:Z): {CI(T;X;Y;Z)}
  CI(T;X:Y): {CI(T;X;Y)}
  CI(T;X:Z): {CI(T;X;Z)}
  CI(T;Y:Z): {CI(T;Y;Z)}
  CI(T;X:Y,X:Z): {CI(T;X;Y,X;Z)}
  CI(T;X:Y,Y:Z): {CI(T;X;Y,Y;Z)}
  CI(T;X:Z,Y:Z): {CI(T;X;Z,Y;Z)}
  CI(T;X:Y,X:Z,Y:Z): {CI(T;X;Y,X;Z,Y;Z)}
  UI(T;X\Y,Z): {UI(T;X\Y,Z)}
  UI(T;Y\X,Z): {UI(T;Y\X,Z)}
  UI(T;Z\X,Y): {UI(T;Z\X,Y)}
  SI(T;X,Y:Z): {SI(T;X,Y;Z)}
  SI(T;Y,X:Z): {SI(T;Y,X;Z)}
  SI(T;Z,X:Y): {SI(T;Z,X;Y)}
  UI(T;X,Y\Z): {UI(T;X,Y\Z)}
  UI(T;X,Z\Y): {UI(T;X,Z\Y)}
  UI(T;Y,Z\X): {UI(T;Y,Z\X)}
  SI(T;X,Y,Z): {SI(T;X,Y,Z)}"""
  print(msg.format(**tripid))

  return 0
#^ main()

main()
