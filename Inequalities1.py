#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:00:59 2024

@author: hugofraser
"""

'''for testing causal inequality violation of various switches, including
white noise, dephasing and depolarising'''

from qutip import *
import numpy as np
import cvxpy as cp

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
    
def Ames(a,x): # Alice measure and reprepare
    if a==0 and x==0:
        return tensor(m00,m00) # a=0, x=0
    elif a==0 and x==1:
        return tensor(m00,m11) # a=0, x=1
    elif a==1 and x==0:
        return tensor(m11,m00)# a=1, x=0
    elif a==1 and x==1:
        return tensor(m11,m11) # a=1, x=1
    elif a==2 and x==2:
        return identity(4)
    
def Bmes(b,y): # Bob measurement in the X or Z directions
    if b==0 and y==0:
        return m00 
    elif b==0 and y==1:
        return 0.5*(m00+m01+m10+m11) # b=0, y=1
    elif b==1 and y==0:
        return m11 # b=1, y=0
    elif b==1 and y==1:
        return 0.5*(m00-m01-m10+m11) # b=1, y=1
    elif b==2 and y==2: 
        return identity(2)
    
def Cmes(c,z): # Charlie measurement in the X+Z or the X-Z directions
    if c==0 and z==0:
        return const1*((3+2*(2**0.5))*m00+(1+2**0.5)*m01+(1+2**0.5)*m10+m11)
                        # c=0, z=0
    elif c==0 and z==1:
        return const1*((3+2*(2**0.5))*m00+(-1-2**0.5)*m01+(-1-2**0.5)*m10+m11)
                        # c=0, z=1
    elif c==1 and z==0:
        return const2*((3-2*(2**0.5))*m00+(1-2**0.5)*m01+(1-2**0.5)*m10+m11)
                        # c=1, z=0
    elif c==1 and z==1:
        return const2*((3-2*(2**0.5))*m00+(-1+2**0.5)*m01+(-1+2**0.5)*m10+m11)
                        # c=1, z=1
    elif c==2 and z==2:
        return identity(2)

def Switness(a1,a2,b,c,x1,x2,y,z): #updated causal witness
    return tensor(Ames(a1,x1),Ames(a2,x2),Cmes(c,z),Bmes(b,y))

def ineq6(W):
    val1, val2, val3 = 0,0,0
    n = 8
    for i in range(2**n, 2**(n+1)):
        string = bin(i)[3:]
        a1 = int(string[0])
        a2 = int(string[1])
        b = int(string[2])
        c = int(string[3])
        x1 = int(string[4])
        x2 = int(string[5])
        y = int(string[6])
        z = int(string[7])
        if b==0 and a2 == x1 and y==0:
            val1 += (0.125*(Switness(a1,a2,b,c,x1,x2,y,z)*W)).tr()
        if b==1 and a1 == x2 and y==0:
            val2 += (0.125*(Switness(a1,a2,b,c,x1,x2,y,z)*W)).tr()
        if (b+c)%2 == y*z and x2 == 0 and x1 ==0:
            val3 += (0.25*(Switness(a1,a2,b,c,x1,x2,y,z)*W)).tr()
    return val1+val2+val3

def total(W):
    val1 = 0
    n = 8
    for i in range(2**n, 2**(n+1)):
        string = bin(i)[3:]
        a1 = int(string[0])
        a2 = int(string[1])
        b = int(string[2])
        c = int(string[3])
        x1 = int(string[4])
        x2 = int(string[5])
        y = int(string[6])
        z = int(string[7])
        val1 += ((Switness(a1,a2,b,c,x1,x2,y,z)*W).tr())*0.5**(4)
    return val1

def nosig1(W):
    n = 8
    truth = True
    for i in range(2**n, 2**(n+1)):
        string = bin(i)[3:]
        a1 = int(string[0])
        a2 = int(string[1])
        c = int(string[2])
        x1 = int(string[3])
        x2 = int(string[4])
        y = int(string[5])
        yp = int(string[6])
        z = int(string[7])
        val1 = ((Switness(a1,a2,0,c,x1,x2,y,z)*W).tr())
        val2 = ((Switness(a1,a2,1,c,x1,x2,y,z)*W).tr())
        val3 = ((Switness(a1,a2,0,c,x1,x2,yp,z)*W).tr())
        val4 = ((Switness(a1,a2,1,c,x1,x2,yp,z)*W).tr())
        if abs((val1+val2)-(val3+val4))>=1e-8:
            print(val1+val2)
            print(val3+val4)
            truth = False
    return truth

def nosig2(W):
    n = 8
    truth = True
    for i in range(2**n, 2**(n+1)):
        string = bin(i)[3:]
        b = int(string[0])
        x1 = int(string[1])
        x1p = int(string[2])
        x2 = int(string[3])
        x2p = int(string[4])
        y = int(string[5])
        z = int(string[6])
        zp = int(string[7])
        val1 = ((Switness(0,0,b,0,x1,x2,y,z)*W).tr())
        val2 = ((Switness(0,0,b,1,x1,x2,y,z)*W).tr())
        val3 = ((Switness(0,1,b,0,x1,x2,y,z)*W).tr())
        val4 = ((Switness(0,1,b,1,x1,x2,y,z)*W).tr())
        val5 = ((Switness(1,0,b,0,x1,x2,y,z)*W).tr())
        val6 = ((Switness(1,0,b,1,x1,x2,y,z)*W).tr())
        val7 = ((Switness(1,1,b,0,x1,x2,y,z)*W).tr())
        val8 = ((Switness(1,1,b,1,x1,x2,y,z)*W).tr())
        val9 = ((Switness(0,0,b,0,x1p,x2p,y,zp)*W).tr())
        val10 = ((Switness(0,0,b,1,x1p,x2p,y,zp)*W).tr())
        val11 = ((Switness(0,1,b,0,x1p,x2p,y,zp)*W).tr())
        val12 = ((Switness(0,1,b,1,x1p,x2p,y,zp)*W).tr())
        val13 = ((Switness(1,0,b,0,x1p,x2p,y,zp)*W).tr())
        val14 = ((Switness(1,0,b,1,x1p,x2p,y,zp)*W).tr())
        val15 = ((Switness(1,1,b,0,x1p,x2p,y,zp)*W).tr())
        val16 = ((Switness(1,1,b,1,x1p,x2p,y,zp)*W).tr())
        LH = val1+val2+val3+val4+val5+val6+val7+val8
        RH = val9+val10+val11+val12+val13+val14+val15+val16
        if abs(LH-RH)>=1e-12:
            truth = False
            print(LH)
            print(RH)
    return truth

def iden(a):
    return identity(a)*(1/a)
    

maxmix = tensor(basis(2,0),basis(2,0))+tensor(basis(2,1),basis(2,1)) # choi state
m00 = Qobj([[1,0],[0,0]])
m01 = Qobj([[0,1],[0,0]]) 
m10 = Qobj([[0,0],[1,0]]) 
m11 = Qobj([[0,0],[0,1]]) 
const1 = 1/(4+2*(2**0.5)) # constants 1 and 2 are for charlie's measurement
const2 = 1/(4-2*(2**0.5))

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
    
W_extended = 0.5*ptrace(ket2dm(process),[0,1,2,3,5,6])

print('                     ineq 6: %.5f'%(ineq6(W_extended)))
print('total = ',total(W_extended))

'''define white noise'''

W_white = (1/16)*identity([2,2,2,2,2,2])


print('         ineq 6 white noise: %.5f'%(ineq6(W_white)))

'''define dephased switch'''

for i in range(4):
    process_p1 = tensor(m00,mats(i),identity(2),m00,m00)
    process_p2_unordered = tensor(m00,mats(i),identity(2),m11,m11)
    process_p2 = process_p2_unordered.permute([2,3,0,1,4,5])
    if i == 0:
        process = process_p1+process_p2
    else:
        process+= process_p1+process_p2
        
W_dephased = 0.5*process


'''define depolarised switch'''

for i in range(4):
    process_p1 = tensor(m00,mats(i),identity(2))
    process_p2_unordered = tensor(m00,mats(i),identity(2))
    process_p2 = process_p2_unordered.permute([2,3,0,1])
    if i == 0:
        process = process_p1+process_p2
    else:
        process+= process_p1+process_p2
        
W_depolarised = 0.5*tensor(process,identity(2)/2,identity(2)/2)



print('            ineq 6 dephased: %.5f'%(ineq6(W_dephased)))

'''calculation for mixed switch'''

print('         ineq 6 depolarised: %.5f'%(ineq6(W_depolarised)))

'''calculation for mixed switch'''

v = 0.8786796564

W_mixed = v*W_extended + (1-v)*W_white
