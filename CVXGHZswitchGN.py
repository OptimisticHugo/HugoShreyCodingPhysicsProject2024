#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:15:04 2024

@author: hugofraser
"""

'''testing cvx code on GHZ causal matrix for general white noise'''

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
                          ,kets(string[1]),kets(string[1]),kets(0),kets(0),kets(0))
    process_p2_unordered = tensor(kets(0),kets(string[0]),kets(string[0])
                          ,kets(string[1]),kets(string[1]),kets(1),kets(1),kets(1))
    process_p2 = process_p2_unordered.permute([2,3,0,1,4,5,6,7])
    if i == 2**n:
        process = process_p1+process_p2
    else:
        process+= process_p1+process_p2
    
W_extended_qobj = 0.5*ptrace(ket2dm(process),[0,1,2,3,5,6,7])
W_extended = np.array(W_extended_qobj)

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

SwitchFlipper = np.kron(AliceBobFlipper,np.identity(8))

AliceBobFlipper2 = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]])

SwitchFlipper2 = np.kron(np.identity(2),
                         np.kron(AliceBobFlipper2,np.identity(4)))
SwitchFlipper3 = np.kron(np.identity(2),AliceBobFlipper2)
SwitchFlipper4 = np.kron(AliceBobFlipper2,np.identity(2))

iden_0 = np.identity(64)*(1/16)

# initialise variables
W_A1A2BCD = cp.Variable((128,128),hermitian=True)
W_A2A1BCD = cp.Variable((128,128),hermitian=True)
W_gen = cp.Variable((128,128), hermitian=True)

W_A2A1BCD_reordered = SwitchFlipper @ W_A2A1BCD @ SwitchFlipper 
W_gen_reordered = SwitchFlipper @ W_gen @ SwitchFlipper 

# tracing out causal order
Di_W_A1A2BC = cp.partial_trace(W_A1A2BCD,(2,2,2,2,2,2,2),6) 
CiDi_W_A1A2B = cp.partial_trace(Di_W_A1A2BC,(2,2,2,2,2,2),5)
BiCiDi_W_A1A2 = cp.partial_trace(CiDi_W_A1A2B,(2,2,2,2,2),4)
A2oBiCiDi_W_A1A2i = cp.partial_trace(BiCiDi_W_A1A2,(2,2,2,2),3)
A2iA2oBiCiDi_W_A1 = cp.partial_trace(A2oBiCiDi_W_A1A2i,(2,2,2),2)
A1oA2iA2oBiCiDi_W_A1i = cp.partial_trace(A2iA2oBiCiDi_W_A1,(2,2),1)

Di_W_A2A1BC = cp.partial_trace(W_A2A1BCD,(2,2,2,2,2,2,2),6)
CiDi_W_A2A1B = cp.partial_trace(Di_W_A2A1BC,(2,2,2,2,2,2),5)
BiCiDi_W_A2A1 = cp.partial_trace(CiDi_W_A2A1B,(2,2,2,2,2),4)
A1oBiCiDi_W_A2A1i = cp.partial_trace(BiCiDi_W_A2A1,(2,2,2,2),3)
A1iA1oBiCiDi_W_A2 = cp.partial_trace(A1oBiCiDi_W_A2A1i,(2,2,2),2)
A2oA1iA1oBiCiDi_W_A2i = cp.partial_trace(A1iA1oBiCiDi_W_A2,(2,2),1)

Di_W_gen_A1A2BC = cp.partial_trace(W_gen,(2,2,2,2,2,2,2),6)
CiDi_W_gen_A1A2B = cp.partial_trace(Di_W_gen_A1A2BC,(2,2,2,2,2,2),5)
BiCiDi_W_gen_A1A2 = cp.partial_trace(CiDi_W_gen_A1A2B,(2,2,2,2,2),4)
A2oBiCiDi_W_gen_A1A2i = cp.partial_trace(BiCiDi_W_gen_A1A2,(2,2,2,2),3)
A2iA2oBiCiDi_W_gen_A1 = cp.partial_trace(A2oBiCiDi_W_gen_A1A2i,(2,2,2),2)
A1oA2iA2oBiCiDi_W_gen_A1i = cp.partial_trace(A2iA2oBiCiDi_W_gen_A1,(2,2),1)

A1oA2oBiCiDi_W_gen_A1iA2i = cp.partial_trace(A2oBiCiDi_W_gen_A1A2i,(2,2,2),1)
A1oA2oBiCiDi_W_gen_A1iA2iI = cp.kron(A1oA2oBiCiDi_W_gen_A1iA2i,iden(2))
A1oA2oBiCiDi_W_gen_A1iIA2i = SwitchFlipper3 @ A1oA2oBiCiDi_W_gen_A1iA2iI @ SwitchFlipper3

Di_W_gen_A2A1BC = cp.partial_trace(W_gen_reordered,(2,2,2,2,2,2,2),6)
CiDi_W_gen_A2A1B = cp.partial_trace(Di_W_gen_A2A1BC,(2,2,2,2,2,2),5)
BiCiDi_W_gen_A2A1 = cp.partial_trace(CiDi_W_gen_A2A1B,(2,2,2,2,2),4)
A1oBiCiDi_W_gen_A2A1i = cp.partial_trace(BiCiDi_W_gen_A2A1,(2,2,2,2),3)
A1iA1oBiCiDi_W_gen_A2 = cp.partial_trace(A1oBiCiDi_W_gen_A2A1i,(2,2,2),2)
A2oA1iA1oBiCiDi_W_gen_A2i = cp.partial_trace(A1iA1oBiCiDi_W_gen_A2,(2,2),1)

A1oBiCiDi_W_gen_A2A1iI = cp.kron(A1oBiCiDi_W_gen_A2A1i,iden(2))
A1oBiCiDi_W_gen_A1iIA2 = AliceBobFlipper @ A1oBiCiDi_W_gen_A2A1iI @ AliceBobFlipper



constraints = [W_extended + W_gen == W_A1A2BCD+W_A2A1BCD_reordered]
constraints += [W_gen >> 0]
constraints += [W_A1A2BCD >> 0]
constraints += [W_A2A1BCD >> 0]



# constraints on W_A1A2BCD
constraints += [cp.kron(BiCiDi_W_A1A2,iden(8))==
                cp.kron(A2oBiCiDi_W_A1A2i,iden(16))]

constraints += [cp.kron(A2iA2oBiCiDi_W_A1,iden(32))==
                cp.kron(A1oA2iA2oBiCiDi_W_A1i,iden(64))] #


# constraints on W_A2A1BCD
constraints += [cp.kron(BiCiDi_W_A2A1,iden(8))==
                cp.kron(A1oBiCiDi_W_A2A1i,iden(16))]

constraints += [cp.kron(A1iA1oBiCiDi_W_A2,iden(32))==
                cp.kron(A2oA1iA1oBiCiDi_W_A2i,iden(64))] #

# constraints on W_gen (to be adapted!!!!!!!)
constraints += [cp.kron(A2iA2oBiCiDi_W_gen_A1,iden(32))==
                cp.kron(A1oA2iA2oBiCiDi_W_gen_A1i,iden(64))]
constraints += [cp.kron(A1iA1oBiCiDi_W_gen_A2,iden(32))==
                cp.kron(A2oA1iA1oBiCiDi_W_gen_A2i,iden(64))]
constraints += [cp.kron(BiCiDi_W_gen_A1A2,iden(8)) == 
                cp.kron(A2oBiCiDi_W_gen_A1A2i,iden(16))+
                cp.kron(A1oBiCiDi_W_gen_A1iIA2,iden(8))-
                cp.kron(A1oA2oBiCiDi_W_gen_A1iIA2i,iden(16))]


# solve CVX problem

objective = cp.Minimize(cp.trace(W_gen))
prob = cp.Problem(objective,constraints)
prob.solve()

# Print result.
result_r = ((prob.value).real)/4

print("first test: The minimum value for r is {}".format(result_r))