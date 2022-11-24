#OA Authors: StAefanos Papanikolaou <stefanos.papanikolaou@mail.wvu.edu>
# BSD 2-Clause License
# Copyright (c) 2019, PapStatMechMat
# All rights reserved.
# How to cite SeaPy:
# S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

#from dcnn_prediction import *
import os
import matplotlib.pyplot as plt
from scipy import *
import numpy as np
import matplotlib as mpl

mpl.rc('lines', linewidth=1, color='black')
mpl.rc('font', size=16,family='serif')
mpl.rc('text',color='black')
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['xtick.minor.size']=4
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['ytick.minor.size']=4
mpl.rcParams['grid.linewidth']=2.0
mpl.rcParams['axes.labelsize']=28
mpl.rcParams['legend.fontsize']=20

mtype=['o','s','>','<','^','v','p','*','h','D','x','H','.']
ltype=['-','--','-.','-','--','-.','--','-','-.']
col=['b','g','r','c','m','y']


def SmartAdd(Total,A):
    # If lines of A less than Total then copy only the shape of A
    sA=array(shape(A))
    sT=array(shape(Total))
    print(sA,sT,'--shapes')
    if (sA==sT).all():
        return Total + A
    else:
        if sA[0]<sT[0]:
            return Total[:sA[0],:] + A
        else:
            return Total[:,:] + A[:sT[0],:]



def RunNeuralNetOnData(RepeatTraining):
#Run dCNN for imaging recognition across classes                                                                                                                                            
    if RepeatTraining=='True':
        os.system('python train.py --train Training/ --num_classes 2')
    else:
        print('Importing training model...')

    Targets=glob.glob('datasets/Testing/*/*.*.jpg' )
    print(Targets)
    trainpath='datasets/Training/'
    P_ellips=[]
    P_circ=[]
    Indexes=[]
    os.system('rm CumulativeWeights.txt')
    os.system('rm CumulativeDataTypes.txt')
    for target in Targets:
        print(target)
        #CumulativeDataTypes is stored here...
        os.system('python predict.py --trainpath datasets/Training --filepath '+target)
    DataWeights=loadtxt('CumulativeWeights.txt')
    fo=open('CumulativeDataTypes.txt')
    ls2=fo.readlines()
    fo.close()
    return ls2 , DataWeights

def PlotWeights(Weights):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(Weights[:,0],Weights[:,1],'o',alpha=0.6)
    ax.set_xlabel('Type 1')
    ax.set_ylabel('Type 2')
    fig.tight_layout()
    fig.savefig('CumulativeWeights.png',transparent=True)
    plt.close(fig)
    return None

def PlotLibAverages(Weights,Types,dirLib,Labels):
    fsLib=glob.glob(dirLib+'/*.txt')
    dirComp='LibComparisonFigs'
    figref=plt.figure()
    axref=figref.add_subplot(111)
    os.system('mkdir '+dirComp)
    DataLib=[]
    ComparisonData=[]
    dir0=os.getcwd()
    i=0
    print(dir0)

    for label in Labels:
        print(label)
        for f in fsLib:
            print(f)
            if label in f: # Only one dataset with distinct label should be present
                d_label_av=loadtxt(dir0+'/'+f)
                DataLib.append(d_label_av)
        # d_av calculation
    print(DataLib,'DataLib')
    for i,typ in enumerate(Types):
        for label in Labels:
            if label in typ:
                dir_ref=typ.split(label)[0]            
                Case=typ.split(label+'.')[1].split('.jpg')[0] # Training images need be jpg
                os.chdir(dir_ref+label+'/Runs/data_txtfiles')
                f_Actual=glob.glob('CumDataLib_*'+label+'_*'+'Case-'+Case+'*.txt')[0]
                print(f_Actual)
                d_Actual=loadtxt(f_Actual) # Found the actual dataset for the given typ... Now, build the prediction of the given dataset... w0 * d_av0 + w1 * d_av1 then store
                axref.plot(d_Actual[:,1],d_Actual[:,2]) #Strain-Stress
                w_labels=Weights[i,:]
                for j,label2 in enumerate(Labels):
                    w=w_labels[j]
                    d_label_av=DataLib[j]
                    if j==0:
                        d_av = w * d_label_av.copy()
                    else:
                        d_av=SmartAdd(d_av , w * d_label_av)                
                ComparisonData.append([Case,label,d_av,d_Actual])
                os.chdir(dir0)

    os.chdir(dir0+'/'+dirComp)
    for cl, l, dav, dact in ComparisonData:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(dav[:,1],dav[:,2],c='blue') #Strain-Stress
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Stress (Pa)')
        ax.plot(dact[:,1],dact[:,2],'o',c='red') #Strain-Stress
        fig.tight_layout()
        fig.savefig('Strain-Stress-Comparison-Case'+cl+'_'+'label'+l+'.png')
        plt.close(fig)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(dav[:,1],dav[:,3],c='blue') #Strain-Damage
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Damage')
        ax.plot(dact[:,1],dact[:,3],'o',c='red') #Strain-Damage
        fig.tight_layout()
        fig.savefig('Strain-Damage-Comparison-Case'+cl+'_'+'label'+l+'.png')
        plt.close(fig)
    axref.set_xlabel('Strain (%)')
    axref.set_ylabel('Stress (Pa)')
    figref.tight_layout()
    figref.savefig('AllStrainStressFigures.png')
    plt.close(figref)
    return None



def RunDataCollectionAndImages():
    dir0=os.getcwd()
    if os.path.exists('datasets/Testing'):
        dir_labels=glob.glob('datasets/Testing/*')
        labels_list_Test=[]
        ImNames_list_Test=[]
        Fnames_list_Test=[]
        for i in range(len(dir_labels)):
            labels_list_Test.append(dir_labels[i].split('datasets/Testing/')[1].split('/')[0])
            os.chdir(dir_labels[i]+'/Runs')
            os.system('python '+dir0+'/collect_data.py ')
            os.system('python '+dir0+'/make_movies.py')
            os.chdir(dir0)            
            Fnames_Test=glob.glob(dir_labels[i]+'/Runs/data_txtfiles/'+'*.txt')
            ImNames_Test=glob.glob(dir_labels[i]+'/Runs/data_imgfiles/'+'*.jpg')
            ImNames_list_Test.extend(ImNames_Test)
            Fnames_list_Test.extend(Fnames_Test)
    else:
        sys.exit('Please Place the Testing Directory inside datasets')
    if os.path.exists('datasets/Training'):
        dir_labels=glob.glob('datasets/Training/*')
        print(dir_labels,'dir_labels')
        labels_list_Train=[]
        ImNames_list_Train=[]
        Fnames_list_Train=[]
        for i in range(len(dir_labels)):
            labels_list_Train.append(dir_labels[i].split('datasets/Training/')[1].split('/')[0])
            os.chdir(dir_labels[i]+'/Runs')
            os.system('python '+dir0+'/collect_data.py ')
            os.system('python '+dir0+'/make_movies.py')
            os.chdir(dir0)
            Fnames_Train=glob.glob(dir_labels[i]+'/Runs/data_txtfiles/'+'*.txt')
            ImNames_Train=glob.glob(dir_labels[i]+'/Runs/data_imgfiles/'+'*.jpg')
            ImNames_list_Train.extend(ImNames_Train)
            Fnames_list_Train.extend(Fnames_Train)
    else:
        sys.exit('Please Place the Training Directory inside datasets')
    return ImNames_list_Test, ImNames_list_Train, Fnames_list_Test, Fnames_list_Train, labels_list_Test, labels_list_Train

def RunLibraryStorage(fnames,labels):
    print(fnames)
    os.system('mkdir LibraryStorage')
    Cases=[]
    Datasets=[]
    for f in fnames:
        if 'CumDataLib' in f: # stress-strain curve
            label=f.split('_Training_')[1].split('_Case')[0] # Works only for training files... Training must be on filename
            Case=f.split('Case-')[1].split('_')[0]
            Cases.append([Case,label])
            a=loadtxt(f)
            Datasets.append(a)
            print(f)
    # Averaging
    Datasets_av=[]        
    for i,label in enumerate(labels):
        num=0
        for c, l in Cases:
            if l==label:
                d_index=Cases.index([c,l])
                d=Datasets[d_index]
                if num==0:
                    Datasets_av.append(d)
                else:
                    Datasets_av[i]=SmartAdd(Datasets_av[i],d)
                num+=1
        if num>0:
            Datasets_av[i]/=float(num)
        # Save
        savetxt('LibraryStorage/LibData_'+label+'.txt',Datasets_av[i])
    return None


def RunSEA(fs,labs):
    # For every Case, generate a set of eigenmodes (fixed in the RunSEAmodes file)
    # The eigenmodes form one image!
    import RunSEAmodes as Sea
    print('RunSEAmodes about to run...')
    N=len(fs)
    D0=Sea.BuildDataMatrix(Zs_test)
    e_pred,d_pred=Sea.Perform_and_PredictFuture(abs(D0),e_full,s)
    figesd,axesd,axtesd=Sea.MakeStressStrainPlot(s_test, e_test, s_full, e_full, e_pred,inc_full0)

    return None



def FindDataSets():
    if os.path.exists('datasets/Testing'):
        dir0=os.getcwd()
        dir_labels=glob.glob('datasets/Testing/*')
        labels_list_Test=[]
        for i in range(len(dir_labels)):
            labels_list_Test.append(dir_labels[i].split('datasets/Testing/')[1].split('/')[0])
            os.chdir(dir_labels[i])
            Fnames_Test=glob.glob(dir_labels[i]+'/'+'*.txt')
            ImNames_Test=glob.glob(dir_labels[i]+'/'+'*.jpg')
            os.chdir(dir0)
    else:
        sys.exit('Please Place the Testing Directory inside datasets')
    if os.path.exists('datasets/Training'):
        dir0=os.getcwd()
        dir_labels=glob.glob('datasets/Training/*')
        labels_list_Train=[]
        for i in range(len(dir_labels)):
            labels_list_Train.append(dir_labels[i].split('datasets/Training/')[1].split('/')[0])
            os.chdir(dir_labels[i])
            Fnames_Train=glob.glob(dir_labels[i]+'/'+'*.txt')
            ImNames_Train=glob.glob(dir_labels[i]+'/'+'*.jpg')
            os.chdir(dir0)
    else:
        sys.exit('Please Place the Training Directory inside datasets')
    return ImNames_Test, ImNames_Train, Fnames_Test, Fnames_Train, labels_list_Test, labels_list_Train



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
    plt.close(fig41)
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
        print(str(i)+'--predicted...'+str(t2[i]))
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
