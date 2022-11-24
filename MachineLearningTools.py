# Authors: Stefanos Papanikolaou <stephanos.papanikolaou@gmail.com>
# BSD 2-Clause License
# Copyright (c) 2021, PapStatMechMat
# All rights reserved.

import ImageTools as IT
import AutomationTools as AT
import unsupervised_learning_tools as UMLT
import pylab as plt
import sys,os,glob
from numpy import shape,array,argsort

def PrepareFeatureMatrix(fs,ftype):
    dt=[]
    print(fs,'fs')
    for i, f_i in enumerate(fs):
        inc=i
        print(inc,f_i)
        if ftype in ['.dat','.txt','.dat2']:
            a3=UMLT.ReadArray(f_i)
        else: # image
            print(f_i)
            img=IT.ReadImage(f_i)
            print(img)
            a3=IT.ImgToScalArr(img)
        print(a3)
        a4=UMLT.preprocess(a3)
        dt.append(a4)
    return dt

def BuildDataMatrix(fs,labs,ftype):
    dt=[]
    for i, f_i in enumerate(fs):
        inc=i
        print(inc,f_i)
        if ftype in ['.dat','.txt','.dat2']:
            a3=UMLT.ReadArray(f_i)
        else: # image
            print(f_i)
            img=IT.ReadImage(f_i)
            print(img)
            a3=IT.ImgToScalArr(img)
        print(a3)
        #a4=UMLT.preprocess(a3)
        s=shape(a3)
        a4=a3.flatten()
        dt.append(a4)
        eps=[]
        for lab in labs:
            eps.append(float(lab))
        eps=array(eps)
        i_eps=argsort(eps)
    return array(dt)[i_eps[:]],s,eps[i_eps[:]]


def SaveTxtFile(x,y,fname):
    """
    save in column format two vectors that may include text entries
    """
    fo=open(fname,'w')
    for x0,y0 in zip(x,y):
        fo.writelines(str(x0)+' '+str(y0)+' '+'\n')
    fo.close()
    return None


def PlotPCA(x,y,per,labs,in_dir,out_dir):
    out_lab='_ID_'+in_dir.split('/')[-1]
    fig_pca1=plt.figure()
    ax_pca1=fig_pca1.add_subplot(111)
    fig_pca2=plt.figure()
    ax_pca2=fig_pca2.add_subplot(111)
    fig_pca3=plt.figure()
    ax_pca3=fig_pca3.add_subplot(111)
    fig_pca4=plt.figure()
    ax_pca4=fig_pca4.add_subplot(111)
    ax_pca1.plot(x,y,'o')
    SaveTxtFile(x,y,out_dir+'/cum-pca1_Total'+out_lab+'.txt')
    ax_pca2.plot(range(1,len(per)+1),per,'o--')
    x2=range(1,len(per)+1)
    SaveTxtFile(x2,per,out_dir+'/cum-pca2_Total'+out_lab+'.txt')
    ax_pca3.plot(labs,y,'o')
    SaveTxtFile(labs,y,out_dir+'/cum-pca3_Total'+out_lab+'.txt')
    ax_pca3.set_xlabel('FileLabelNumbers')
    ax_pca3.set_ylabel('Component 2 Proj')
    ax_pca4.plot(labs,x,'o')
    SaveTxtFile(labs,x,out_dir+'/cum-pca4_Total'+out_lab+'.txt')
    ax_pca4.set_xlabel('FileLabelNumbers')
    ax_pca4.set_ylabel('Component 1 Proj')
    ax_pca2.set_xlabel('number of components (Total)')
    ax_pca2.set_ylabel('cumulative explained variance')
    ax_pca1.set_xlabel('Component 1 Proj.')
    ax_pca1.set_ylabel('Component 2 Proj.')
    fig_pca1.tight_layout()
    fig_pca2.tight_layout()
    fig_pca3.tight_layout()
    fig_pca4.tight_layout()
    fig_pca2.savefig(out_dir+'/cum-pca2_Total'+out_lab+'.png')
    fig_pca1.savefig(out_dir+'/cum-pca1_Total'+out_lab+'.png')
    fig_pca3.savefig(out_dir+'/cum-pca3_Total'+out_lab+'.png')
    fig_pca4.savefig(out_dir+'/cum-pca4_Total'+out_lab+'.png')
    plt.show()
    return None

def PlotWavelets(x,y,labs,in_dir,out_dir):
    out_lab='_ID_'+in_dir.split('/')[-1]
    fig_wl=plt.figure()
    ax_wl=fig_wl.add_subplot(111)
    if len(x)==len(y):
        y0=y
    else:
        y0=y[:-1]
    ax_wl.plot(x,y0,'--o')
    SaveTxtFile(x,y0,out_dir+'/wavelet'+out_lab+'.txt')
    fig_wl.tight_layout()
    fig_wl.savefig(out_dir+'/wavelet'+out_lab+'.png')
    plt.show()
    return None


    
