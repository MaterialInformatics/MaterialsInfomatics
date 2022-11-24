# Authors: Stefanos Papanikolaou <stephanos.papanikolaou@gmail.com>
# BSD 2-Clause License
# Copyright (c) 2021, PapStatMechMat
# All rights reserved.

import ImageTools as IT
import AutomationTools as AT
import unsupervised_learning_tools as UMLT
import pylab as plt
import glob,sys,os

in_dir,out_dir=AT.GetDirectories()
fs=glob.glob(in_dir+'/*.png')

fig_pca1=plt.figure()
ax_pca1=fig_pca1.add_subplot(111)
fig_pca2=plt.figure()
ax_pca2=fig_pca2.add_subplot(111)
fig_pca3=plt.figure()
ax_pca3=fig_pca3.add_subplot(111)
dt=[]
for i, f_i in enumerate(fs):
    inc=i
    print(inc)
    img=IT.ReadImage(f_i)
    a3=IT.ImgToScalArr(img)
    a4=UMLT.preprocess(a3)
    dt.append(a4)
x,y,per=UMLT.ApplyPCA(dt,out_dir,'Total_')
ax_pca1.plot(x,y)
ax_pca2.plot(range(1,len(per)+1),per)

ax_pca2.set_xlabel('number of components (Total)')
ax_pca2.set_ylabel('cumulative explained variance')
ax_pca1.set_xlabel('Component 1 Proj.')
ax_pca1.set_ylabel('Component 2 Proj.')
fig_pca1.tight_layout()
fig_pca2.tight_layout()
fig_pca2.savefig(out_dir+'/cum-pca2_Total.png')
fig_pca1.savefig(out_dir+'/cum-pca1_Total.png')
