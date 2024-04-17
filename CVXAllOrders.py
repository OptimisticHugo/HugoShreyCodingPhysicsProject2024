#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:22:51 2024

@author: hugofraser
"""


'''testing cvx code on extended causal matrix, with moving bob around
i expect that this should not make any difference'''

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

'''define process matrix for extended setup'''

n = 2
for i in range(2**n, 2**(n+1)):
    string = bin(i)[3:]
    process_p1 = tensor(kets(0),kets(string[0]),kets(string[0])
                          ,kets(string[1]),kets(string[1]),kets(0),kets(0))
    process_p2_unordered = tensor(kets(0),kets(string[0]),kets(string[0])
                          ,kets(string[1]),kets(string[1]),kets(1),kets(1))
    process_p2 = process_p2_unordered.permute([2,3,0,1,4,5,6])
    if i == 2**n:
        process = process_p1+process_p2
    else:
        process+= process_p1+process_p2
    
W_extended = np.array(0.5*ptrace(ket2dm(process),[0,1,2,3,5,6]))

'''define dephased switch'''

for i in range(4):
    process_p1 = tensor(m00,mats(i),identity(2),m00,m00)
    process_p2_unordered = tensor(m00,mats(i),identity(2),m11,m11)
    process_p2 = process_p2_unordered.permute([2,3,0,1,4,5])
    if i == 0:
        process = process_p1+process_p2
    else:
        process+= process_p1+process_p2
        
W_dephased = np.array(0.5*process)


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

SwitchFlipper = np.kron(AliceBobFlipper,np.identity(4))

AliceBobFlipper2 = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]])

SwitchFlipper2 = np.kron(np.identity(2),
                         np.kron(AliceBobFlipper2,np.identity(4)))

W_white = np.identity(64)*(1/16)
r = cp.Variable()
objective = cp.Minimize(r)

# initialise variables
W_A1A2CB = cp.Variable((64,64),hermitian =True)
W_A1A2BC_flipper = np.kron(np.identity(16),AliceBobFlipper2)
W_A1A2BC = W_A1A2BC_flipper @ W_A1A2CB @ W_A1A2BC_flipper
W_A1A2iBA2oC_flipper = np.kron(np.identity(8),
                           np.kron(AliceBobFlipper2,np.identity(2)))
W_A1A2iBA2oC = W_A1A2iBA2oC_flipper @ W_A1A2BC @ W_A1A2iBA2oC_flipper
W_A1BA2C_flipper = np.kron(np.identity(4),
                           np.kron(AliceBobFlipper2,np.identity(4)))
W_A1BA2C = W_A1BA2C_flipper @ W_A1A2iBA2oC @ W_A1BA2C_flipper 

W_A1iBA1oA2C_flipper = np.kron(np.identity(2),
                           np.kron(AliceBobFlipper2,np.identity(8)))
W_A1iBA1oA2C = W_A1iBA1oA2C_flipper @ W_A1BA2C @ W_A1iBA1oA2C_flipper
W_BA1A2C_flipper = np.kron(AliceBobFlipper2,np.identity(16))
W_BA1A2C = W_BA1A2C_flipper @ W_A1iBA1oA2C @ W_BA1A2C_flipper

W_A2A1CB = cp.Variable((64,64),hermitian=True)
W_A2A1BC = W_A1A2BC_flipper @ W_A2A1CB @ W_A1A2BC_flipper 
#W_A2BA1C = W_A1BA2C_flipper @ W_A2A1BC @ W_A1BA2C_flipper
#W_BA2A1C = W_BA1A2C_flipper @ W_A2BA1C @ W_BA1A2C_flipper


W_A2A1CB_reordered = SwitchFlipper @ W_A2A1CB @ SwitchFlipper 
W_A2A1BC_reordered = SwitchFlipper @ W_A2A1BC @ SwitchFlipper 
#W_A2BA1C_reordered = SwitchFlipper @ W_A2BA1C @ SwitchFlipper 
#W_BA2A1C_reordered = SwitchFlipper @ W_BA2A1C @ SwitchFlipper 

'''WA1A2'''
Bi_W_A1A2C = cp.partial_trace(W_A1A2CB,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_A1A2 = cp.partial_trace(Bi_W_A1A2C,(2,2,2,2,2),4)
A2oCiBi_W_A1A2i = cp.partial_trace(CiBi_W_A1A2,(2,2,2,2),3)
A2iA2oCiBi_W_A1 = cp.partial_trace(A2oCiBi_W_A1A2i,(2,2,2),2)
A1oA2iA2oCiBi_W_A1i = cp.partial_trace(A2iA2oCiBi_W_A1,(2,2),1)


Ci_W_A1A2B = cp.partial_trace(W_A1A2BC,(2,2,2,2,2,2),5) 
BiCi_W_A1A2 = cp.partial_trace(Ci_W_A1A2B,(2,2,2,2,2),4)
A2oBiCi_W_A1A2i = cp.partial_trace(BiCi_W_A1A2,(2,2,2,2),3)
A2iA2oBiCi_W_A1 = cp.partial_trace(A2oBiCi_W_A1A2i,(2,2,2),2)
A1oA2iA2oBiCi_W_A1i = cp.partial_trace(A2iA2oBiCi_W_A1,(2,2),1)

Ci_W_A1BA2 = cp.partial_trace(W_A1BA2C,(2,2,2,2,2,2),5) 
A2oCi_W_A1BA2i = cp.partial_trace(Ci_W_A1BA2,(2,2,2,2,2),4)
A2iA2oCi_W_A1B = cp.partial_trace(A2oCi_W_A1BA2i,(2,2,2,2),3)
BiA2iA2oCi_W_A1 = cp.partial_trace(A2iA2oCi_W_A1B,(2,2,2),2)
A1oBiA2iA2oCi_W_A1i = cp.partial_trace(BiA2iA2oCi_W_A1,(2,2),1)

Ci_W_BA1A2 = cp.partial_trace(W_BA1A2C,(2,2,2,2,2,2),5) 
A2oCi_W_BA1A2i = cp.partial_trace(Ci_W_BA1A2,(2,2,2,2,2),4)
A2iA2oCi_W_BA1 = cp.partial_trace(A2oCi_W_BA1A2i,(2,2,2,2),3)
A1oA2iA2oCi_W_BA1i = cp.partial_trace(A2iA2oCi_W_BA1,(2,2,2),2)
A1iA1oA2iA2oCi_W_B = cp.partial_trace(A1oA2iA2oCi_W_BA1i,(2,2),1)

'''WA2A1'''

Bi_W_A2A1C = cp.partial_trace(W_A2A1CB,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_A2A1 = cp.partial_trace(Bi_W_A2A1C,(2,2,2,2,2),4) 
A1oCiBi_W_A2A1i = cp.partial_trace(CiBi_W_A2A1,(2,2,2,2),3)
A1iA1oCiBi_W_A2 = cp.partial_trace(A1oCiBi_W_A2A1i,(2,2,2),2)
A2oA1iA1oCiBi_W_A2i = cp.partial_trace(A1iA1oCiBi_W_A2,(2,2),1)

Ci_W_A2A1B = cp.partial_trace(W_A2A1CB,(2,2,2,2,2,2),5) 
BiCi_W_A2A1 = cp.partial_trace(Ci_W_A2A1B,(2,2,2,2,2),4) 
A1oBiCi_W_A2A1i = cp.partial_trace(BiCi_W_A2A1,(2,2,2,2),3)
A1iA1oBiCi_W_A2 = cp.partial_trace(A1oBiCi_W_A2A1i,(2,2,2),2)
A2oA1iA1oBiCi_W_A2i = cp.partial_trace(A1iA1oBiCi_W_A2,(2,2),1)

Ci_W_A2BA1 = cp.partial_trace(W_A2A1CB,(2,2,2,2,2,2),5) 
A1oCi_W_A2BA1i = cp.partial_trace(Ci_W_A2BA1,(2,2,2,2,2),4) 
A1iA1oCi_W_A2B = cp.partial_trace(A1oCi_W_A2BA1i,(2,2,2,2),3)
BiA1iA1oCi_W_A2 = cp.partial_trace(A1iA1oCi_W_A2B,(2,2,2),2)
A2oBiA1iA1oCi_W_A2i = cp.partial_trace(BiA1iA1oCi_W_A2,(2,2),1)

Ci_W_BA2A1 = cp.partial_trace(W_A2A1CB,(2,2,2,2,2,2),5) 
A1oCi_W_BA2A1i = cp.partial_trace(Ci_W_BA2A1,(2,2,2,2,2),4) 
A1iA1oCi_W_BA2 = cp.partial_trace(A1oCi_W_BA2A1i,(2,2,2,2),3)
A2oA1iA1oCi_W_BA2i = cp.partial_trace(A1iA1oCi_W_BA2,(2,2,2),2)
A2iA2oA1iA1oCi_W_B = cp.partial_trace(A2oA1iA1oCi_W_BA2i,(2,2),1)

constraints = [r>=0, r<=5]
constraints += [W_extended+r*W_white == W_A1A2CB+W_A2A1CB_reordered]
constraints += [W_extended+r*W_white == W_A1A2BC+W_A2A1BC_reordered]

constraints += [W_A1A2CB >> 0]
constraints += [W_A1A2BC >> 0]
#constraints += [W_A1BA2C >> 0]
#constraints += [W_BA1A2C >> 0]

constraints += [W_A2A1CB >> 0]
constraints += [W_A2A1BC >> 0]
#constraints += [W_A2BA1C >> 0]
#constraints += [W_BA2A1C >> 0]


'''it is dubious whether to include constraints marked #'''


# constraints on W_A1A2CB
constraints += [cp.kron(CiBi_W_A1A2,iden(4))==cp.kron((A2oCiBi_W_A1A2i),iden(8))]

constraints += [cp.kron(A2iA2oCiBi_W_A1,iden(16))==
                cp.kron(A1oA2iA2oCiBi_W_A1i,iden(32))] #

# constraints on W_A1A2BC
constraints += [cp.kron(BiCi_W_A1A2,iden(4))==cp.kron((A2oBiCi_W_A1A2i),iden(8))]
constraints += [cp.kron(A2iA2oBiCi_W_A1,iden(16))==
                cp.kron(A1oA2iA2oBiCi_W_A1i,iden(32))] #

# constraints on W_A1BA2C
#constraints += [cp.kron(Ci_W_A1BA2,iden(2))==cp.kron((A2oCi_W_A1BA2i),iden(4))]
#constraints += [cp.kron(BiA2iA2oCi_W_A1,iden(16))==
#                cp.kron(A1oBiA2iA2oCi_W_A1i,iden(32))] #

# constraints on W_BA1A2C
#constraints += [cp.kron(Ci_W_BA1A2,iden(2))==cp.kron((A2oCi_W_BA1A2i),iden(4))]
#constraints += [cp.kron(A2iA2oCi_W_BA1,iden(16))==
#                cp.kron(A1oA2iA2oCi_W_BA1i,iden(32))] #



# constraints on W_A2A1CB
constraints += [cp.kron(CiBi_W_A2A1,iden(4))==cp.kron(A1oCiBi_W_A2A1i,iden(8))]
constraints += [cp.kron(A1iA1oCiBi_W_A2,iden(16))==
                cp.kron(A2oA1iA1oCiBi_W_A2i,iden(32))] #

# constraints on W_A2A1BC
constraints += [cp.kron(BiCi_W_A2A1,iden(4))==cp.kron(A1oBiCi_W_A2A1i,iden(8))]
constraints += [cp.kron(A1iA1oBiCi_W_A2,iden(16))==
                cp.kron(A2oA1iA1oBiCi_W_A2i,iden(32))] #

# constraints on W_A2BA1C
#constraints += [cp.kron(Ci_W_A2BA1,iden(4))==cp.kron(A1oCi_W_A2BA1i,iden(8))]
#constraints += [cp.kron(BiA1iA1oCi_W_A2,iden(16))==
#                cp.kron(A2oBiA1iA1oCi_W_A2i,iden(32))] #

# constraints on W_BA2A1C
#constraints += [cp.kron(Ci_W_BA2A1,iden(4))==cp.kron(A1oCi_W_BA2A1i,iden(8))]
#constraints += [cp.kron(A1iA1oCi_W_BA2,iden(16))==
#                cp.kron(A2oA1iA1oCi_W_BA2i,iden(32))] #

prob = cp.Problem(objective,constraints)
prob.solve()

# Print result.

print("The minimum value for r is {:0.11f}".format(prob.value))

print("The minimum value for v is {:0.11f}".format(1/(1+prob.value)))