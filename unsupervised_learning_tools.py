import matplotlib.mlab as mlab
import pylab as plt
import sys,os,glob
from numpy import *
import matplotlib as mpl
from scipy.stats import gaussian_kde
import re,sys
import pywt
import pywt.data
from PIL import Image

def FindNum(string,ftype):
    r=''.join([n for n in string if n.isdigit()])    
    if len(r)>0:
        return r
    else:
        sys.exit("The filename needs to be numbered!")
        return None
def ReadArray(file):
    fo=open(file)
    ls=fo.readlines()
    try:
        a=array([[float(i) for i in l.split(',')] for l in ls])
    except:
        a=array([[float(i) for i in l.split(' ')] for l in ls])
    fo.close()
    return a

def preprocess(a):
    data=a.flatten()
    scaled_data = abs(data - data.mean())/(std(data)+1e-10)
    return scaled_data


def ApplyPCA(dt,dir0,lab):
    from sklearn.decomposition import PCA
    b=array(dt)
    pca = PCA(n_components=3)
    pca.fit(b)
    principalComponents = pca.components_
    dt_pca=pca.fit_transform(b)
    #percent=pca.explained_variance_
    #fig=plt.figure()
    #ax=fig.add_subplot(111)
    #ax.scatter(dt_pca[:,0],dt_pca[:,1],alpha=0.5)
    #ax.set_xlabel('Component 1')
    #ax.set_ylabel('Component 2')
    #fig.tight_layout()
    #fig.savefig(dir0+'/PCAmap_'+lab+'Fig5.png')
    #plt.close(fig)
    #print(shape(dt_pca))
    #fig=plt.figure()
    #ax=fig.add_subplot(111)
    per=pca.explained_variance_ratio_
    var=pca.explained_variance_
    #ax.plot(range(1,len(per)+1),np.cumsum(per),'o--')
    #ax.set_xlabel('number of components')
    #ax.set_ylabel('cumulative explained variance')
    #fig.savefig(dir0+'/PCAratio_'+lab+'_Fig5.png')
    #plt.close(fig)
    x=-sign(dt_pca[0,0])*dt_pca[:,0]/sqrt(var[0])
    y=-sign(dt_pca[0,1])*dt_pca[:,1]/sqrt(var[1])
    pern=cumsum(pca.explained_variance_ratio_)
    return x,y,pern

def ApplyWavelets(dt,labs,dir0,lab):
    """
    set of image vectors at various parameters
    Wavelet operation and tracking fluctuations along parameter direction
    To be stored in dir0 with significance labeled as lab
    """
    b=array(dt)
    img=Image.fromarray(b)
    coeffs2=pywt.dwt2(img,'db2')
    LL,(LH,HL,HH) = coeffs2
    ymean=array([average(abs(HL[j,:]-HL[0,:])) for j in range(len(HL[:,0]))])
    x2=labs[1::2]
    y2=ymean[:-1]
    return x2,y2   


def ArrayReplace(a,t,q):
    s=shape(a)
    a2=[]
    a3=[]
    all_projx=[]
    for i in range(s[0]):
        a2.append([])
        a3.append([])        
        for j in range(s[1]):
            a2[i].append(t[int(a[i,j])-1])
            a3[i].append(q[int(a[i,j])-1])
            all_projx.append(t[int(a[i,j])-1])    
    return array(a2),all_projx,array(a3)

def PlotSaveHistogram(t):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    n, bins, patches = ax.hist(array(t), 18,histtype='bar', edgecolor='black',facecolor='green', alpha=0.75)
    ax.set_xlabel('Grain orientation projection on loading axis X')
    ax.set_ylabel('Number of grains')
    ax.set_xlim((-1,1))
    fig.tight_layout()
    fig.savefig('Fig2_OrientationHist.png')
    plt.close(fig)
    return None

def ArrayToImageSave(b,name):
    from PIL import Image
    s=shape(b)
    a=zeros((s[0],s[1],3), 'uint8')
    max_r=max(b[:,:,0].flatten())
    max_g=max(b[:,:,1].flatten())
    max_b=max(b[:,:,2].flatten())
    a[...,0]=256-int8(b[...,0]*256/max_r)
    a[...,1]=256-int8(b[...,1]*256/max_r)
    a[...,2]=256-int8(b[...,2]*256/max_r)
    img=Image.fromarray(a)
    img.save(name)
    return None

def Save2DImage(z,f, fkind,cmin,cmax,bar,dir0):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    mpb=ax1.imshow(z)
    plt.axis('off')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    #colorbar_ax=fig.add_axes([0.7,0.1,0.05,0.8])
    if bar=='yes':
        fig.colorbar(mpb)
    #plt.clim(cmin,cmax)
    fig.savefig(dir0+'/'+fkind+'_Lib_'+f+'.jpg', bbox_inches='tight', pad_inches = 0, transparent=True)
    plt.close(fig)
    return fig,ax1

def PlotScatterDataVsOrientations(x,y,typ,dir0):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x,y,'o',markerfacecolor=None,alpha=0.25)
    fig.tight_layout()
    fig.savefig(dir0+'/Scatter_'+typ+'_Orientations_Fig4.png')    
    plt.close(fig)
    return None
