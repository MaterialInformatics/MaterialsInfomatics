# Authors: Stefanos Papanikolaou <stefanos.papanikolaou@mail.wvu.edu>
# BSD 2-Clause License
# Copyright (c) 2019, PapStatMechMat
# All rights reserved.
# How to cite SeaPy:
# S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

def PickMode(Ep,col,s):
    p0=Ep.reshape(s)
    rp0=real(p0)
    return rp0

# considers a sequence of strain images (text arrays or true images) and produces eigenmodes, which are then stitched together.
def fingerprint_from_txt(fs):
    # sequence assumed ordered
    # filenames assumed to contain invariant of interest
    Z=[]
    for i,f in enumerate(fs):
        A=loadtxt(f)
        if i==0:
            s=shape(A)
        Z.append(A.flatten())
    D=Z.T
    X=D[:,:-1]
    Y=D[:,1:]
    U2,Sig2,Vh2 = svd(X, False)
    
    r = 20
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]

    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig))
    mu,W = eig(Atil)

    Phi = dot(dot(dot(Y, V), inv(Sig)), W)

    Finger0=PickMode(Phi,0,s)
    Finger1=PickMode(Phi,1,s)
    Finger2=PickMode(Phi,2,s)
    Finger3=PickMode(Phi,0,s)
    
    return [Finger0,Finger1,Finger2,Finger3]



    



    

    
