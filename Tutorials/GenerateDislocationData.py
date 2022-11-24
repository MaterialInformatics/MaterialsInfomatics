from scipy import *
import sys,os,glob
import pylab as plt

Npix=128
#####################################################################
#Draw strain fields of I1 on lattice for consecutive time-steps
# The thing that changes in the example is the x-coordinate
# Initiate dislocation pair at +x0 , -x0
# Then, x = x0 + edot * t
# t = arange(0, tmax), tmax=edot/x0
# sxx = - D * y * (3*x**2 + y**2) / (x**2 + y**2 + eps)**2
# syy =   D * y * (x**2 - y**2) / (x**2 + y**2 + eps)**2
# sxy = syx = D * x * (x**2 - y**2) / (x**2 + y**2)**2
# szz = nu * (sxx + syy)
# szz = szx = syz = szy = 0
# D = G * b / (2*pi * (1-nu))
# I1 = 2* (1+nu) * D/3 * (y/(x**2 + y**2))
# Organize 2, 4, 8, 16 dislocation examples...
# Process: Nucleate at random increments, then propagate them at fixed rate.
# Find Eigenmodes
# Organize images and recognize 2, 4, 8 and 16 dislocation strain fields...
#####################################################################

"""
os.system('rm *.png')
if os.path.exists('datasets'):
    os.system('rm -rf datasets')
os.system('mkdir datasets')
"""
if len(sys.argv)!=4:
    sys.exit('python GenerateDislocationData.py Nucleation/Glide Training/Testing NumSamples')


NucleationORGlide=sys.argv[1]
TrainingORTesting=sys.argv[2]

sigma_rate = 2. #Thrs are between 0 and 1 / Time evolves from 0 to 1
v = 1.
a=.05

Ns=10. # size of area in Burgers vectors
NumRandomCases=int(sys.argv[3])

nu=0.33 #same as copper
b=3.e-10
G=1.e11

epsilon=1e-2
### x, y are in units of b  ###
I1 = lambda x, y: - (1-2*nu)*nu/(2*pi*(1-nu)) * (y / (x**2 + y**2+epsilon))
###############################

D = G * b / ( 2 * pi * (1-nu) )

Nums=2**arange(0,2,1) # Number of sources in the system

time=linspace(0,1,200) # when time crosses threshold disl-nucleates and both disls start moving at fixed rate.


Xv=linspace(-Ns,Ns,Npix)

Yv=linspace(-Ns,Ns,Npix)

Xfield,Yfield=meshgrid(Xv,Yv)

#Plot stress field
def PlotStressTest():
    Ifield=zeros((128,128))
    for i in range(128):
        for j in range(128):
            Ifield[i,j]+=I1(Xfield[i,j],Yfield[i,j]) 
    print(max(Ifield.flatten()), 'max')
    print(min(Ifield.flatten()),'min')

    plt.imshow(Ifield)
    plt.colorbar()
    plt.show()
    sys.exit()
    return None

#PlotStressTest()

def FillImage(Im, xts,yts,sign_b):
    Im2=Im.copy()
    s=shape(Im2)
    for xt,yt in zip(xts,yts):        
        if xt==0 and yt==0:
            # source is null
            Im2=Im2
        else:
            for i in range(s[0]):
                for j in range(s[1]):                
                    x=Xfield[i,j]-xt
                    y=Yfield[i,j]-yt          

                    Im2[i,j]+=I1(x,y)*sign_b
    
    return Im2

dpi=100
sigma_ext=0.
for case in range(NumRandomCases):
    for num in Nums:

        Thrs=random.random(num)
        print(Thrs)

        X0s=(2.*random.random(num)- 1.) * max(Xv) / 4.
        Y0s=(2.*random.random(num)- 1.) * max(Yv) / 4.

        #print(X0s,Y0s)
        for i,t in enumerate(time):  
              
            Ifield=zeros((Npix,Npix))
            sigma_ext = sigma_rate * t
            Xs1 = array([ X0s[j] - max(a , v*(sigma_ext > Thrs[j]) ) * t for j in range(len(X0s)) ])   #*Nucs
            Ys1 = Y0s #* Thrs
            Ifield = FillImage(Ifield,Xs1,Ys1,+1)

            if NucleationORGlide=='Nucleation':
                Xs2 = array([ X0s[j] + max(a , v*(sigma_ext > Thrs[j]) ) * t for j in range(len(X0s)) ])   #*Nucs
                Ys2 = Y0s #* Thrs
                Ifield = FillImage(Ifield,Xs2,Ys2,-1)
            
            #print(sigma_ext,Thrs,Xs1-X0s,'sigma-Thrs-dX')

            fig=plt.figure()
            ax1=fig.add_subplot(111)
            ax1.axis('off')
            mpb=ax1.imshow(Ifield,alpha=0.9)            
            mpb.set_clim(-.01,.01)            
            #plt.colorbar(mpb)
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])

            fig.savefig('Im_Case'+str(case).rjust(2,'0')+'_NumDislocations'+str(num).rjust(3,'0')+'_InvariantSnapshot'+str(i).rjust(4,'0')+'.png',bbox_inches='tight',pad_inches=0,transparent=True,dpi=dpi)
            plt.close(fig)

            savetxt('I-Field-Data_Case'+str(case).rjust(2,'0')+'_NumDislocations'+str(num).rjust(3,'0')+'_'+str(i).rjust(4,'0')+'.txt',Ifield)

        os.system('mkdir datasets/'+TrainingORTesting)
        os.system('mkdir datasets/'+TrainingORTesting+'/'+NucleationORGlide)
        dirt='datasets/'+TrainingORTesting+'/'+NucleationORGlide+'/Case'+str(case).rjust(3,'0')+'_' + NucleationORGlide + '_N'+str(num).rjust(3,'0')

        os.system('mkdir '+dirt)
        os.system('mv Im_Case*Dislocations*.png I-Field*.txt '+dirt)
        
        
        print(dirt)
        

