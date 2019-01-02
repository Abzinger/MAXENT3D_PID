# test_and_gate.py
from MAXENT3D_PID import pid, MAXENT3D_PID_Exception


# AND DUBLICATE
andDgate = dict()
andDgate[ (0,0,0,0) ] = .25
andDgate[ (0,0,1,0) ] = .25
andDgate[ (0,1,0,1) ] = .25
andDgate[ (1,1,1,1) ] = .25

# ECOS parameters 
parms = dict()
parms['max_iters'] = 100

print("Starting MAXENT3D_PID.pid() on AND gate.")
try:
  sol = pid(andDgate, parallel='on', output=3, **parms)
  msg="""Synergistic information: {CI}
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
  print(msg.format(**sol))
except MAXENT3D_PID_Exception:
  print("Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")

print("The End")
