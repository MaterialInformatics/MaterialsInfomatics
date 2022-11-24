# Authors: Stefanos Papanikolaou <stefanos.papanikolaou@mail.wvu.edu>
# BSD 2-Clause License
# Copyright (c) 2019, PapStatMechMat
# All rights reserved.
# How to cite SeaPy:
# S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

from scipy import *


def first_invariant3D(F):
    return F[:,0]+F[:,4]+F[:,8]

def first_invariant2D(F):
    return F[:,0]+F[:,3]

def second_principal_invariant3D(F):
    # (F11 F12 F13) #
    # (F21 F22 F23) #
    # (F31 F32 F33) #
    return F[:,0] * F[:,4] + F[:,4] * F[:,8] + F[:,0] * F[:,8] - F[:,1] * F[:,3] - F[:,5] * F[:,7] - F[:,2] * F[:,6]

def second_principal_invariant2D(F):
    # (F11 F12) #
    # (F21 F22) #    
    return F[:,0] * F[:,3] - F[:,1] * F[:,2]

def second_main_invariant3D(F):
    return first_invariant3D(F)**2 - 2 * second_principal_invariant3D(F)

def third_principal_invariant3D(F):
    # (F11 F12 F13) #                                                                                       
    # (F21 F22 F23) #                                                                                       
    # (F31 F32 F33) #  
    return - F[:,2] * F[:,4] * F[:,6] + F[:,1] * F[:,5] * F[:,6] +\
        F[:,2] * F[:,3] * F[:,7]-\
        F[:,0] * F[:,5] * F[:,7]-\
        F[:,1] * F[:,3] * F[:,8]+\
        F[:,0] * F[:,3] * F[:,8]

def third_main_invariant3D(F):
    I1=first_invariant3D(F)
    I2=second_principal_invariant3D(F)
    I3=third_principal_invariant3D(F)
    return I1**3 - 3 * I1 * I2 + 3 * I3





