# Authors: Stefanos Papanikolaou <stephanos.papanikolaou@gmail.com>
# BSD 2-Clause License
# Copyright (c) 2021, PapStatMechMat
# All rights reserved.
import ImageTools as IT
import AutomationTools as AT
import unsupervised_learning_tools as UMLT
import MachineLearningTools as ML
import pylab as plt
import glob,sys,os
import fingerprint_tools as FT
import RunSEAmodes as Sea
from numpy import *

in_dir,out_dir,method_choice, file_choice=AT.GetDirectories()
filetype=file_choice
max_num=80

print(file_choice)

if method_choice == 'PCA Analysis':
    print('PCA Analysis','method_choice')
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    print('GotFiles',fs,labs)
    dt=ML.PrepareFeatureMatrix(fs,filetype)
    print(dt,'dt')
    x,y,per=UMLT.ApplyPCA(dt,out_dir,'Total_')
    ML.PlotPCA(x,y,per,labs,out_dir)
    
if method_choice=='WL Analysis':
    print('WL Analysis','method_choice')
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    print('GotFiles',fs,labs)
    dt=ML.PrepareFeatureMatrix(fs,filetype)
    print(dt,'dt')
    x,y=UMLT.ApplyWavelets(dt,labs,out_dir,'Total_')
    ML.PlotWavelets(x,y,labs,out_dir)                            

if method_choice == 'SEA Analysis':
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    D0,s,labfloats=ML.BuildDataMatrix(fs,labs,filetype)
    e_pred,d_pred=Sea.Perform_and_PredictFuture(abs(D0),labfloats,s,out_dir)
    #figesd,axesd,axtesd=Sea.PlotEims(e_pred,d_pred)

if method_choice == 'Correlations:2D':
    import spatial_correlation_tools as SC
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    D0,s,labfloats=ML.BuildDataMatrix(fs,labs,filetype)
    corrs=SC.Correlations2D(D0,s)
    SC.Plot2DCorrelations(D0,corrs,labs,out_dir,s)

if method_choice == 'ConstructAtomicConfiguration':
    choice2_1,choice2_2 = AT.GetOptionsConstrConfig()

