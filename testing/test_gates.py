# test_gates.py
from MAXENT3D_PID import pid, MAXENT3D_PID_Exception


# AND gate
andgate = dict()
andgate[ (0,0,0,0) ] = .125
andgate[ (0,0,0,1) ] = .125
andgate[ (0,0,1,0) ] = .125
andgate[ (0,1,0,0) ] = .125
andgate[ (0,0,1,1) ] = .125
andgate[ (0,1,0,1) ] = .125
andgate[ (0,1,1,0) ] = .125
andgate[ (1,1,1,1) ] = .125

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


# Parallel off 

print("And gate: ", pid(andgate, output=0))
print("\n...\n")
print("And Duplicate gate: ", pid(andDgate, output=0))
print("\n...\n")
print("Xor gate: ", pid(xorgate, output=0))
print("\n...\n")
print("Xor Duplicate gate: ", pid(xorDgate, output=0))
print("\n...\n")
print("Xor Multi core gate: ", pid(xorMgate, output=0))
print("\n...\n")
print("Xor Loss gate: ", pid(xorLgate, output=0))
print("\n                   =================================================================== \n")

# Parallel on
print("Parallel on") 
print("And gate: ", pid(andgate, parallel='on'))
print("\n...\n")
print("And Duplicate gate: ", pid(andDgate, parallel='on'))
print("\n...\n")
print("Xor gate: ", pid(xorgate, parallel='on'))
print("\n...\n")
print("Xor Duplicate gate: ", pid(xorDgate,parallel='on'))
print("\n...\n")
print("Xor Multi core gate: ", pid(xorMgate,parallel='on'))
print("\n...\n")
print("Xor Loss gate: ", pid(xorLgate,parallel='on'))

