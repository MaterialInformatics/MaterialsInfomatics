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

in_dir,out_dir,method_choice, file_choice=AT.GetDirectories()
filetype=file_choice
max_num=200
print(file_choice)
if method_choice == 'PCA Analysis':
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    dt=ML.PrepareFeatureMatrix(fs,filetype)
    x,y,per=UMLT.ApplyPCA(dt,out_dir,'Total_')
    ML.PlotPCA(x,y,per,labs,out_dir)
if method_choice == 'Image Correlations':
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    all_colors=IT.FindColorsInImage(fs)
    2DCorrelations=IT.FindCorrelations(fs,all_colors)
    3DCorrelations=IT.2Dto3D(2Dcorrelations)
    method_choice2,f_to3D=AT.3DreconstructionQuestion()
    if method_choice2=='Yes':
        3Dmicrostructure = IT.GrowthProcess3D(f_to3D,box,all_colors)
        LAMMPSInputFile  = IT.BuildMolecularDynamicsFile(3Dmicrostructure,params)
        submitLAMMPSsimulation = IT.SubmitLAMMPS(LAMMPSInputFile)

    
