import sys,os,glob
from dcnn_prediction import *
import pylab as plt
from scipy import *

if len(sys.argv)!=4:
    sys.exit("python RunDislocationTutorial.py True/False True/False NumImages")
Generate=sys.argv[1]
RepeatTraining=sys.argv[2]
NumImages=N=sys.argv[3]
dir0=os.getcwd()
print(dir0)
if Generate=='True':
    os.system('rm *.png')
    if os.path.exists('datasets'):
        os.system('rm -rf datasets')
    os.system('mkdir datasets')
    print('Dislocation Data will be generated from scratch...')

    #Generate training and testing data
    os.system('python GenerateDislocationData.py Nucleation Training '+N)
    os.system('python GenerateDislocationData.py Glide Training '+N)
    os.system('python GenerateDislocationData.py Nucleation Testing '+N)
    os.system('python GenerateDislocationData.py Glide Testing '+N)
    print('RunSEAmodes about to run...')

    #Calculate predominant EIMs and copy them in the main directories
    os.system('python runSEAmodes.py Nucleation Training')
    os.system('python runSEAmodes.py Glide Training')
    os.system('python runSEAmodes.py Nucleation Testing')
    os.system('python runSEAmodes.py Glide Testing')

#Run dCNN for imaging recognition across classes
if RepeatTraining==True:
    os.system('python train.py --train Training/ --val Testing/ --num_classes 2')
else:
    print('Importing training model...')

Targets=glob.glob('datasets/Testing/*/*.*.jpg' )
trainpath='datasets/Training/'
P_nucl=[]
P_glid=[]
Indexes=[]
for target in Targets:
    print(target)
    Weights=MakePrediction(target,trainpath)[0]
    Indexes.append(str(target))
    
    P_nucl.append(float(Weights[0]))
    P_glid.append(float(Weights[1]))

print('Saving Probability Data into Text File...')
savetxt('Weights_NuclGrid_NoIndexes.txt',transpose(array([P_nucl,P_glid])))
write_file('Weights_NuclGlid.txt',Indexes,[P_nucl,P_glid])

print('Plotting Now...')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(range(len(P_nucl)),P_nucl,'o',label=r'$P_{nucleation}$')
ax.plot(range(len(P_nucl)),P_glid,'s',label=r'$P_{glide}$')
ax.set_xlabel('Testing Sample Index')
ax.set_ylabel('Probability')
fig.savefig('Plot_WeightsIndex.png')

plt.show()



