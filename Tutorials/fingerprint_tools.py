# Authors: Stefanos Papanikolaou <stefanos.papanikolaou@mail.wvu.edu>
# BSD 2-Clause License
# Copyright (c) 2019, PapStatMechMat
# All rights reserved.
# How to cite SeaPy:
# S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

def ImageFingerPrint(Ep,col,s):
    fig41=plt.figure()
    ax41=fig41.add_subplot(111)
    p0=Ep[:,col].reshape(s)
    rp0=real(p0)
    plt.imshow(rp0/max(rp0.flatten()))
    sc=str(col)
    plt.title(sc+'-th Mode')
    plt.clim(0,1)
    fig41.tight_layout()
    fig41.savefig(sc+'th-Mode.png')
    return None

def CalculateEvolution(Phi,X, t):
    b = dot(pinv(Phi), X[:,0])
    t2 = np.linspace(min(t), FinalInc, 50)
    Psi = np.zeros([r, len(t2)], dtype='complex')
    for i,_t in enumerate(t2):
        Psi[:,i] = multiply(power(mu, _t/dt), b)

    D2 = dot(Phi, Psi)

    sigmaps=[]
    tps=[]
    Eavs=[]
    StrainTot=[]
    for i in range(len(D2[0,:])):
        print str(i)+'--predicted...'+str(t2[i])
        F=D2[:,i]
        eps=0.001*t2[i]   # arbitrarily defined                             
        # Elastic contributions needs to be added back to the stress...     
    #ElasticEnergy = sum(Eslopes[:] * eps)                              
    # Then deliver the average...                                       
        sigma=MakeImagePred(F,i,s,s2,eps)
        tps.append(t2[i])
    #sigmaps.append(sigma+ElasticEnergy)                                
        Eavs.append(np.average(real(F)))
        StrainTot.append(eps)
    return None
