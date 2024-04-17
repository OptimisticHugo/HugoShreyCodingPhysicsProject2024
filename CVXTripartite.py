#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:24:34 2024

@author: hugofraser
"""

'''testing cvx code on tripartite causal matrix we want r = 1.576 for random
robustness'''

# Import packages.
import cvxpy as cp
import numpy as np
from qutip import *

def kets(i): 
    if int(i) == 0:
        return ket('0')
    elif int(i) == 1:
        return ket('1')
    
def mats(i):
    if i == 0:
        return tensor(m00,m00)
    elif i == 1:
        return tensor(m01,m01)
    elif i == 2:
        return tensor(m10,m10)
    elif i == 3:
        return tensor(m11,m11)

maxmix = tensor(basis(2,0),basis(2,0))+tensor(basis(2,1),basis(2,1)) # choi state
m00 = Qobj([[1,0],[0,0]])
m01 = Qobj([[0,1],[0,0]]) 
m10 = Qobj([[0,0],[1,0]]) 
m11 = Qobj([[0,0],[0,1]]) 

'''generate tripartite process matrix'''
n = 2
for i in range(2**n, 2**(n+1)):
    string = bin(i)[3:]
    process_p1 = tensor(kets(0),kets(string[0]),kets(string[0])
                          ,kets(string[1]),kets(string[1]),kets(0))
    process_p2_unordered = tensor(kets(0),kets(string[0]),kets(string[0])
                          ,kets(string[1]),kets(string[1]),kets(1))
    process_p2 = process_p2_unordered.permute([2,3,0,1,4,5])
    if i == 2**n:
        process = process_p1+process_p2
    else:
        process+= process_p1+process_p2
    
W_trip_qobj = 0.5*ptrace(ket2dm(process),[0,1,2,3,5])
W_trip = np.array(W_trip_qobj)

def iden(a):
    return np.identity(a)*(1/a)

AliceBobFlipper = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])

SwitchFlipper = np.kron(AliceBobFlipper,np.identity(2))

AliceBobFlipper2 = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]])

SwitchFlipper2 = np.kron(np.identity(2),
                         np.kron(AliceBobFlipper2,np.identity(4)))

iden_0 = np.identity(32)*(1/8)
r = cp.Variable()
objective = cp.Minimize(r)

# initialise variables
W_ABC = cp.Variable((32,32),hermitian =True)
W_BAC = cp.Variable((32,32),hermitian=True)
# when adding, WABC and WBAC must have same tensor structure
W_BAC_reordered = SwitchFlipper @ W_BAC @ SwitchFlipper 

Ci_W_AB = cp.partial_trace(W_ABC,(2,2,2,2,2),4) # tracing out causal order
BoCi_W_ABi = cp.partial_trace(Ci_W_AB,(2,2,2,2),3)
BiBoCi_W_A = cp.partial_trace(BoCi_W_ABi,(2,2,2),2)
AoBiBoCi_W_Ai = cp.partial_trace(BiBoCi_W_A,(2,2),1)

Ci_W_BA = cp.partial_trace(W_BAC,(2,2,2,2,2),4) # tracing out causal order
AoCi_W_BAi = cp.partial_trace(Ci_W_BA,(2,2,2,2),3)
AiAoCi_W_B = cp.partial_trace(AoCi_W_BAi,(2,2,2),2)
BoAiAoCi_W_Bi = cp.partial_trace(AiAoCi_W_B,(2,2),1)


constraints = [r>=0, r<=5]
constraints += [W_trip+r*iden_0 == W_ABC + W_BAC_reordered] # change Wm for W+rIden
constraints += [W_ABC >> 0]
constraints += [W_BAC >> 0]

'''it is dubious whether to include constraints marked #'''


# constraints on W_ABC
constraints += [cp.kron(Ci_W_AB,iden(2)) == cp.kron(BoCi_W_ABi,iden(4))]

constraints += [cp.kron(BiBoCi_W_A,iden(8)) == cp.kron(AoBiBoCi_W_Ai,iden(16))]


# constraints on W_BAC
constraints += [cp.kron(Ci_W_BA,iden(2)) == cp.kron(AoCi_W_BAi,iden(4))]

constraints += [cp.kron(AiAoCi_W_B,iden(8)) == cp.kron(BoAiAoCi_W_Bi,iden(16))]

prob = cp.Problem(objective,constraints)
prob.solve()

# Print result.

print("The minimum value for r is {:0.11f}".format(prob.value))

print("The minimum value for v is {:0.11f}".format(1/(1+prob.value)))