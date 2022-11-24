import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power,ones,average
from scipy.signal import convolve2d
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn
import glob,sys,os
from scipy import array,log2,shape, argsort,loadtxt
from numpy.lib.stride_tricks import as_strided as ast
from itertools import product

Rfactor=1
inc_full0=200
inc_test0=140
num_pred=20

import matplotlib as mpl
import numpy as np
from scipy.stats import gaussian_kde


mpl.rc('lines', linewidth=1, color='black')
mpl.rc('font', size=20,family='serif')
mpl.rc('text',color='black')
mpl.rcParams['xtick.major.size']=16
mpl.rcParams['xtick.minor.size']=10
mpl.rcParams['xtick.labelsize']=20
mpl.rcParams['ytick.labelsize']=20
mpl.rcParams['ytick.major.size']=16
mpl.rcParams['ytick.minor.size']=10
mpl.rcParams['grid.linewidth']=2.0
mpl.rcParams['axes.labelsize']=28
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['savefig.dpi']=250


mtype=['o','s','>','<','^','v','p','*','h','D','x','H','.']
ltype=['-','--','-.','-','--','-.','--','-','-.']
col=['b','g','r','c','m','y','brown','cyan','black']
G=1e11

def DislocationState(f):
    
    A=loadtxt(f)
    strain_zz=A.copy()
    sigma_zz =A.copy() * G
    return strain_zz, sigma_zz,shape(A)

def Make2DImageField(z,inc):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    mpb=ax1.imshow(z)
    plt.axis('off')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    #colorbar_ax=fig.add_axes([0.7,0.1,0.05,0.8])
    fig.colorbar(mpb)
    fig.savefig('O_SEA_damage_'+str(inc).rjust(5,'0')+'.png', bbox_inches='tight', pad_inches = 0, transparent=True)
    return fig,ax1

def Make2DImageSigma(z,inc):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    mpb=ax1.imshow(z)
    plt.axis('off')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    #colorbar_ax=fig.add_axes([0.7,0.1,0.05,0.8])
    fig.colorbar(mpb)
    fig.savefig('O_SEA_sigma_'+str(inc).rjust(5,'0')+'.png', bbox_inches='tight', pad_inches = 0, transparent=True)
    return fig,ax1

def Make2DImageStrain(z,inc):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    mpb=ax1.imshow(z)
    plt.axis('off')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    #colorbar_ax=fig.add_axes([0.7,0.1,0.05,0.8])
    fig.colorbar(mpb)
    fig.savefig('O_SEA_'+str(inc).rjust(5,'0')+'.png', bbox_inches='tight', pad_inches = 0, transparent=True)
    return fig,ax1

def Make2DImageTexture(z,inc):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    from matplotlib import cm
    mpb=ax1.imshow(z*10,cmap=cm.gist_ncar,alpha=0.9)
    plt.axis('off')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    #colorbar_ax=fig.add_axes([0.7,0.1,0.05,0.8])
    #fig.colorbar(mpb)
    fig.savefig('O_SEA_'+str(inc).rjust(5,'0')+'.png', bbox_inches='tight', pad_inches = 0, transparent=True)
    return fig,ax1

def MakeStressStrainPlot(s_test, e_test, s_full, e_full, e_pred , inc):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(e_test, s_test, 's',c='blue',alpha=0.75)
    axt=ax.twinx()
    #axt.plot(e_test, d_test, 's',c='red' ,alpha=0.75)
    ax.plot(e_full, s_full, '-' ,c='blue')
    #axt.plot(e_full, d_full, '-',c='maroon',lw=1)
    #ax.plot(e_full, d_full, '-',c='maroon',lw=1,alpha=0.45,label=' ')
    #axt.plot(e_pred, d_pred, '--',c='purple',lw=3,alpha=0.75)
    ax.plot([0], [0], '--',c='purple',lw=3,alpha=0.75,label=' ')
    #from signalsmooth import smooth
    w0=35
    e_pred2=e_full #smooth(e_full,window_len=w0)
    #print len(e_pred2)
    s_pred2=s_full #smooth(s_full,window_len=w0)
    #d_pred2=smooth(d_full,window_len=w0)
    print(len(s_pred2))
    ax.plot([0], [0], '-',c='red',lw=5,alpha=0.4,label=' ')
    ax.plot(e_pred2, s_pred2, '-',c='navy',lw=5,alpha=0.5,label=' ')
    #axt.plot(e_pred2, 0.95*d_pred2, '-',c='red',lw=5,alpha=0.5)
    ax.set_xlabel(r'$\langle\epsilon\rangle$')
    ax.set_ylabel(r'$\langle \sigma_{zz}\rangle$'+'(MPa)')
    axt.set_ylabel(r'$\langle I_{1}^{(\epsilon)}\rangle$')
    """
    ax.set_xlim((0.0,0.0012))
    axt.set_ylim(bottom=-0.005)
    axt.set_xlim((0.0,0.0012))
    ax.set_ylim(bottom=-0.5)
    """
    #ax.set_xticks(ax.get_xticks()[::2])
    axt.spines['right'].set_color('red')
    axt.yaxis.label.set_color('red')
    axt.tick_params(axis='y', colors='red')    
    ax.spines['left'].set_color('blue')
    ax.yaxis.label.set_color('blue')
    l=ax.legend(loc='upper left')
    l.draw_frame(False)
    #l=axt.legend(loc='upper left', bbox_to_anchor=(0., 0.))
    #l.draw_frame(False)
    ax.tick_params(axis='y', colors='blue')
    fig.savefig('O_SEA_'+str(inc).rjust(5,'0')+'.png', bbox_inches='tight', pad_inches = 0, transparent=True)
    #plt.show()
    return fig , ax , axt

def BuildDataMatrix(Dms):
    return Dms

def Energy(p):
    e=array([sum(p[i,:] * p[i,:]) for i in range(len(p[:,0]))])
    return e

def MakeImage(P,col,s1,outd):
    fig41=plt.figure()
    ax41=fig41.add_subplot(111)
    print(col,s1,'makeImage')
    p0=P[:,col].reshape(s1)
    #p0=Energy(p).reshape(s1)                                              
    rp0=real(p0)
    mpb=plt.imshow(rp0/max(rp0.flatten()))
    #plt.clim(0,1e5) # For dislocation examples
    plt.axis('off')
    ax41.set_yticklabels([])
    ax41.set_xticklabels([])
    sc=str(col)
#    fig41.savefig(outd+'/O_'+sc+'th-InvariantMode_NoCBar.png',bbox_inches='tight', pad_inches = 0, transparent=True)
    plt.colorbar(mpb)#,extend='both')
    fig41.savefig(outd+'/O_SEA_'+sc+'th-InvariantMode.png',bbox_inches='tight', pad_inches = 0, transparent=True)
    #plt.title(sc+'-th Damage Mode')    
    #fig=plt.figure()
    #ax=fig.add_subplot(111)
    #colorbar_ax=fig.add_axes([0.7,0.1,0.05,0.8])
    #fig.colorbar(mpb)
    return None

def MakeImagePred(P,col,s1,eps,outd):
    fig41=plt.figure()
    ax41=fig41.add_subplot(111)
    p=P.reshape(s1)
    sav=real(p.flatten().mean())
    p0=p #Energy(p).reshape(s1)                                            
    rp0=real(p0)
    print(rp0.flatten().mean(),rp0.flatten().max())
    mpb=plt.imshow(rp0)
    plt.clim(-.1,.1)
    plt.axis('off')
    ax41.set_yticklabels([])
    ax41.set_xticklabels([])
    sc=str(format(eps,'.0e'))[:]
    #fig41.savefig('O_'+sc+'th-StepInvariant.png',bbox_inches='tight', pad_inches = 0, transparent=True)
    plt.colorbar(mpb)#,extend='both')
    fig41.savefig(outd+'/O_SEA_'+sc+'th-StepStrainInvariant_WithCbar.png',bbox_inches='tight', pad_inches = 0, transparent=True)
    #plt.title(' Damage Field '+r'$\phi$'+' at '+r'$\epsilon=$'+sc)
    
    return sav

def MakePlot_SV(Sig,r,outd):
    ####Plotting
    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    ax2.plot(Sig,'s',markersize=20)
    ax2.set_xlabel('index '+r'$j$')
    ax2.set_ylabel(r'$\varsigma_j$')
    ax2.set_xlim((-0.2,r))
    fig2.tight_layout()
    fig2.savefig(outd+'/O_SEA_SV.png',bbox_inches='tight', pad_inches = 0, transparent=True)
    ############
    return fig2,ax2

def MakePlot_Eigen(mu,outd):
    t0 = np.linspace(0, 2*pi, 20)
    fig3=plt.figure()
    ax3=fig3.add_subplot(111)
    ax3.plot(real(mu),imag(mu),'s',markersize=20)
    ax3.plot(cos(t0), sin(t0),'--')
    ax3.set_xlabel(r'$Re(\mu)$')
    ax3.set_ylabel(r'$Im(\mu)$')
    fig3.tight_layout()
    fig3.savefig(outd+'/O_SEA_Eigen.png',bbox_inches='tight', pad_inches = 0, transparent=True)
    return fig3,ax3,t0

def Predict(Phi,b,mu,s,t,r,outd):
    print(t,'--t')
    dt=t[1]-t[0]
    tmin=min(t)
    tmax=max(t)
    t2 = np.linspace(tmin, tmax, num_pred)
    Psi = np.zeros([r, len(t2)], dtype='complex')    
    for i,_x in enumerate(t2):
        print(_x,'_x')
        print(b,'b')
        print(i)
        print(shape(Psi))
        Psi[:,i] = multiply(power(mu, _x/dt), b)
    # compute DMD reconstruction          
    D2 = dot(Phi, Psi)
    #np.allclose(D, D2) # True 
    sigmaps=[]
    tps=[]
    for i in range(len(D2[0,:])):
        print(str(i)+'--predicted...'+str(t2[i]))
        F=D2[:,i]
        if i==0: #subtract background
            F0=average(F)
        eps=t2[i]
        sigma=MakeImagePred((F-F0),i,s,eps,outd)
        tps.append(t2[i])
        sigmaps.append(sigma+eps)
    return tps,sigmaps

def Perform_and_PredictFuture(D0,eps,s,outd):
    D=D0.T #Data Matrix     
    X=D[:,:-1]
    Y=D[:,1:]
    # SVD of input matrix  
    U2,Sig2,Vh2 = svd(X, False)
    r = 5   # rank-5 truncation
    fig_SV,ax_SV=MakePlot_SV(Sig2,r,outd)
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    # build A tilde                                                        
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig))
    mu,W = eig(Atil)
    fig_Eigen,ax_Eigen,t0=MakePlot_Eigen(mu,outd)
    # build DMD modes                                          
    Phi = dot(dot(dot(Y, V), inv(Sig)), W)
    MakeImage(Phi,0,s,outd)
    MakeImage(Phi,1,s,outd)
    MakeImage(Phi,2,s,outd)
    MakeImage(Phi,3,s,outd)
    # compute time evolution                                        
    b = dot(pinv(Phi), X[:,1])
    tps,sigmaps=Predict(Phi,b,mu,s,eps,r,outd)
    return tps,sigmaps
