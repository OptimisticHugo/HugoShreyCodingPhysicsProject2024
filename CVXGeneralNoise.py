#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:13:31 2024

@author: hugofraser
"""

'''testing cvx code on extended causal matrix for general worst case noise'''

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
    
def iden(a):
    return np.identity(a)*(1/a)

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
    
W_extended_qobj = 0.5*ptrace(ket2dm(process),[0,1,2,3,5,6])
W_extended = np.array(W_extended_qobj)

#print(W_extended_qobj.tr())

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
SwitchFlipper3 = np.kron(np.identity(2),AliceBobFlipper2)
SwitchFlipper4 = np.kron(AliceBobFlipper2,np.identity(2))
SwitchFlipper5 = np.kron(np.identity(16),AliceBobFlipper2)
SwitchFlipper6 = np.kron(np.identity(8),AliceBobFlipper2)
SwitchFlipper7 = np.kron(np.identity(4),np.kron(AliceBobFlipper2,np.identity(2)))

iden_0 = np.identity(64)*(1/16)

# initialise variables
W_A1A2CB = cp.Variable((64,64),hermitian =True)
W_A2A1CB = cp.Variable((64,64),hermitian=True)
W_gen = cp.Variable((64,64), hermitian=True)

W_A2A1CB_reordered = SwitchFlipper @ W_A2A1CB @ SwitchFlipper

Bi_W_gen_A1A2C = cp.partial_trace(W_gen,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_gen_A1A2 = cp.partial_trace(Bi_W_gen_A1A2C,(2,2,2,2,2),4)
A2oCiBi_W_gen_A1A2i = cp.partial_trace(CiBi_W_gen_A1A2,(2,2,2,2),3)
A2iA2oCiBi_W_gen_A1 = cp.partial_trace(A2oCiBi_W_gen_A1A2i,(2,2,2),2)
A1oA2iA2oCiBi_W_gen_A1i = cp.partial_trace(A2iA2oCiBi_W_gen_A1,(2,2),1)

A1oA2oCiBi_W_gen_A1iA2i = cp.partial_trace(A2oCiBi_W_gen_A1A2i,(2,2,2),1)
A1oA2oCiBi_W_gen_A1iA2iI = cp.kron(A1oA2oCiBi_W_gen_A1iA2i,iden(2))
A1oA2oCiBi_W_gen_A1iIA2i = SwitchFlipper3 @ A1oA2oCiBi_W_gen_A1iA2iI @ SwitchFlipper3

W_gen_reordered = SwitchFlipper @ W_gen @ SwitchFlipper

Bi_W_gen_A2A1C = cp.partial_trace(W_gen_reordered,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_gen_A2A1 = cp.partial_trace(Bi_W_gen_A2A1C,(2,2,2,2,2),4)
A1oCiBi_W_gen_A2A1i = cp.partial_trace(CiBi_W_gen_A2A1,(2,2,2,2),3)
A1iA1oCiBi_W_gen_A2 = cp.partial_trace(A1oCiBi_W_gen_A2A1i,(2,2,2),2)
A2oA1iA1oCiBi_W_gen_A2i = cp.partial_trace(A1iA1oCiBi_W_gen_A2,(2,2),1)

A1oCiBi_W_gen_A2A1iI = cp.kron(A1oCiBi_W_gen_A2A1i,iden(2))
A1oCiBi_W_gen_A1iIA2 = AliceBobFlipper @ A1oCiBi_W_gen_A2A1iI @ AliceBobFlipper

'''experimental'''
Ci_W_gen_A1A2B = cp.partial_trace(W_gen,(2,2,2,2,2,2),4)
Ci_W_gen_A1A2BI = cp.kron(Ci_W_gen_A1A2B,iden(2))
Ci_W_gen_A1A2IB = SwitchFlipper5 @Ci_W_gen_A1A2BI@ SwitchFlipper5

A1iCi_W_A1oA2B = cp.partial_trace(Ci_W_gen_A1A2B,(2,2,2,2,2),0)
A1iA2iCi_W_A1oA2oB = cp.partial_trace(A1iCi_W_A1oA2B,(2,2,2,2),1)
A1iA2iCi_W_IA1oA2oB = cp.kron(iden(2),A1iA2iCi_W_A1oA2oB)
A1iA2iCi_W_IA1oA2oBI = cp.kron(A1iA2iCi_W_IA1oA2oB,iden(2))
A1iA2iCi_W_IA1oA2oIB = SwitchFlipper6@ A1iA2iCi_W_IA1oA2oBI @SwitchFlipper6
A1iA2iCi_W_IA1oIA2oB = SwitchFlipper7@ A1iA2iCi_W_IA1oA2oIB @SwitchFlipper7
A1iA2iCiBi_W_IA1oIA2o = cp.partial_trace(A1iA2iCi_W_IA1oIA2oB,(2,2,2,2,2),4)
'''experimantal'''

'''experimental2'''
Bi_W_ex_A1A2C = cp.partial_trace(W_extended,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_ex_A1A2 = cp.partial_trace(Bi_W_ex_A1A2C,(2,2,2,2,2),4)
A2oCiBi_W_ex_A1A2i = cp.partial_trace(CiBi_W_ex_A1A2,(2,2,2,2),3)
A2iA2oCiBi_W_ex_A1 = cp.partial_trace(A2oCiBi_W_ex_A1A2i,(2,2,2),2)
A1oA2iA2oCiBi_W_ex_A1i = cp.partial_trace(A2iA2oCiBi_W_ex_A1,(2,2),1)

W_ex_reordered = SwitchFlipper @ W_extended @ SwitchFlipper

Bi_W_ex_A2A1C = cp.partial_trace(W_ex_reordered,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_ex_A2A1 = cp.partial_trace(Bi_W_ex_A2A1C,(2,2,2,2,2),4)
A1oCiBi_W_ex_A2A1i = cp.partial_trace(CiBi_W_ex_A2A1,(2,2,2,2),3)
A1iA1oCiBi_W_ex_A2 = cp.partial_trace(A1oCiBi_W_ex_A2A1i,(2,2,2),2)
A2oA1iA1oCiBi_W_ex_A2i = cp.partial_trace(A1iA1oCiBi_W_ex_A2,(2,2),1)

Ci_W_ex_A1A2B = cp.partial_trace(W_extended,(2,2,2,2,2,2),4)
Ci_W_ex_A1A2BI = cp.kron(Ci_W_ex_A1A2B,iden(2))
Ci_W_ex_A1A2IB = SwitchFlipper5 @Ci_W_ex_A1A2BI@ SwitchFlipper5

A1iCi_W_ex_A1oA2B = cp.partial_trace(Ci_W_ex_A1A2B,(2,2,2,2,2),0)
A1iA2iCi_W_ex_A1oA2oB = cp.partial_trace(A1iCi_W_ex_A1oA2B,(2,2,2,2),1)
A1iA2iCi_W_ex_IA1oA2oB = cp.kron(iden(2),A1iA2iCi_W_ex_A1oA2oB)
A1iA2iCi_W_ex_IA1oA2oBI = cp.kron(A1iA2iCi_W_ex_IA1oA2oB,iden(2))
A1iA2iCi_W_ex_IA1oA2oIB = SwitchFlipper6@ A1iA2iCi_W_ex_IA1oA2oBI @SwitchFlipper6
A1iA2iCi_W_ex_IA1oIA2oB = SwitchFlipper7@ A1iA2iCi_W_ex_IA1oA2oIB @SwitchFlipper7
A1iA2iCiBi_W_ex_IA1oIA2o = cp.partial_trace(A1iA2iCi_W_ex_IA1oIA2oB,(2,2,2,2,2),4)
'''experimental2'''



Bi_W_A1A2C = cp.partial_trace(W_A1A2CB,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_A1A2 = cp.partial_trace(Bi_W_A1A2C,(2,2,2,2,2),4)
A2oCiBi_W_A1A2i = cp.partial_trace(CiBi_W_A1A2,(2,2,2,2),3)
A2iA2oCiBi_W_A1 = cp.partial_trace(A2oCiBi_W_A1A2i,(2,2,2),2)
A1oA2iA2oCiBi_W_A1i = cp.partial_trace(A2iA2oCiBi_W_A1,(2,2),1)

Bi_W_A2A1C = cp.partial_trace(W_A2A1CB,(2,2,2,2,2,2),5) # tracing out causal order
CiBi_W_A2A1 = cp.partial_trace(Bi_W_A2A1C,(2,2,2,2,2),4)
A1oCiBi_W_A2A1i = cp.partial_trace(CiBi_W_A2A1,(2,2,2,2),3)
A1iA1oCiBi_W_A2 = cp.partial_trace(A1oCiBi_W_A2A1i,(2,2,2),2)
A2oA1iA1oCiBi_W_A2i = cp.partial_trace(A1iA1oCiBi_W_A2,(2,2),1)


constraints = [W_extended + W_gen == W_A1A2CB + W_A2A1CB_reordered]
constraints += [W_gen >> 0]
constraints += [W_A1A2CB >> 0]
constraints += [W_A2A1CB >> 0]



# constraints on W_A1A2CB
constraints += [cp.kron(CiBi_W_A1A2,iden(4))==cp.kron((A2oCiBi_W_A1A2i),iden(8))]

constraints += [cp.kron(A2iA2oCiBi_W_A1,iden(16))==
                cp.kron(A1oA2iA2oCiBi_W_A1i,iden(32))] 


# constraints on W_A2A1CB
constraints += [cp.kron(CiBi_W_A2A1,iden(4))==cp.kron(A1oCiBi_W_A2A1i,iden(8))]

constraints += [cp.kron(A1iA1oCiBi_W_A2,iden(16))==
                cp.kron(A2oA1iA1oCiBi_W_A2i,iden(32))] #

# constraints on W_gen
constraints += [cp.kron(A2iA2oCiBi_W_gen_A1,iden(16))==
                cp.kron(A1oA2iA2oCiBi_W_gen_A1i,iden(32))]
constraints += [cp.kron(A1iA1oCiBi_W_gen_A2,iden(16))==
                cp.kron(A2oA1iA1oCiBi_W_gen_A2i,iden(32))]
constraints += [cp.kron(CiBi_W_gen_A1A2,iden(4)) == 
                cp.kron(A2oCiBi_W_gen_A1A2i,iden(8))+
                cp.kron(A1oCiBi_W_gen_A1iIA2,iden(4))-
                cp.kron(A1oA2oCiBi_W_gen_A1iIA2i,iden(8))]
#constraints += [cp.kron(CiBi_W_gen_A1A2,iden(4))==Ci_W_gen_A1A2IB]
#constraints += [A1iA2iCi_W_IA1oIA2oB==cp.kron(A1iA2iCiBi_W_IA1oIA2o,iden(2))]
constraints += [A1iA2iCi_W_ex_IA1oIA2oB==cp.kron(A1iA2iCiBi_W_ex_IA1oIA2o,iden(2))]

# solve CVX problem

objective = cp.Minimize(cp.trace(W_gen))
prob = cp.Problem(objective,constraints)
prob.solve()

# Print result.
result_r = ((prob.value).real)/4

print("first test: The minimum value for r is {}".format(result_r))