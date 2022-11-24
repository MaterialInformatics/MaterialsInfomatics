import tkinter.filedialog as filedialog
import tkinter as tk
from tkinter import ttk
import glob,sys,os
from unsupervised_learning_tools import FindNum
from numpy import array, argsort
import ImageTools as IT
import AutomationTools as AT
import unsupervised_learning_tools as UMLT
import MachineLearningTools as ML
import pylab as plt
import glob,sys,os
import fingerprint_tools as FT
import RunSEAmodes as Sea


def get_path(filename):
    return os.path.realpath(filename)

def GetFiles(in_dir,in_type,max_num):
    fs=array(glob.glob(in_dir+'/'+'*'+in_type))
    labs=[]
    for i, f_i in enumerate(fs):
        inc=i
        num=FindNum(f_i.split('/')[-1],in_type)
        labs.append(int(num))
    labs=array(labs)
    i_l=argsort(labs)[:]
    if max_num<=len(i_l):
        return fs[i_l[:max_num]], labs[i_l[:max_num]]
    else:
        return fs[i_l[:]], labs[i_l[:]]

    
    
    

def runGUI():

    master = tk.Tk()
    master.title('MATI')

    def clear_all(*args):
        input_entry.delete(0, tk.END)
        output_entry.delete(0, tk.END)

    def submit_run(*args):
        max_num=80
        in_dir=input_entry.get()
        out_dir=output_entry.get()
        method_choice=variables.get()
        file_choice=filetype=variables2.get()
        if method_choice == 'PCA Analysis':
            print('PCA Analysis','method_choice')
            fs,labs=AT.GetFiles(in_dir,filetype,max_num)
            print('GotFiles',fs,labs)
            dt=ML.PrepareFeatureMatrix(fs,filetype)
            print(dt,'dt')
            x,y,per=UMLT.ApplyPCA(dt,out_dir,'Total_')
            ML.PlotPCA(x,y,per,labs,in_dir,out_dir)

        if method_choice=='WL Analysis':
            print('WL Analysis','method_choice')
            fs,labs=AT.GetFiles(in_dir,filetype,max_num)
            print('GotFiles',fs,labs)
            dt=ML.PrepareFeatureMatrix(fs,filetype)
            print(dt,'dt')
            x,y=UMLT.ApplyWavelets(dt,labs,out_dir,'Total_')
            ML.PlotWavelets(x,y,labs,in_dir,out_dir)
                                
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
        
    def input():
        global inp
        input_path = tk.filedialog.askdirectory()
        input_entry.delete(1, tk.END)  # Remove current text in entry
        input_entry.insert(0, input_path)  # Insert the 'path'
        inp=input_path
    
    def output():
        global outp
        path = tk.filedialog.askdirectory()
        output_entry.delete(1, tk.END)  # Remove current text in entry
        output_entry.insert(0, path)  # Insert the 'path'
        outp=path

    top_frame = tk.Frame(master)
    bottom_frame = tk.Frame(master)
    line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')

    input_path = tk.Label(top_frame, text="Input Directory:")
    input_entry = ttk.Entry(top_frame, text="",width=60)
    browse1 = tk.Button(top_frame, text="Browse", command=input)
    input_dir=input_entry.get()

    output_path = tk.Label(bottom_frame, text="Output Directory:")
    output_entry = ttk.Entry(bottom_frame, text="",width=60)
    
    browse2 = tk.Button(bottom_frame, text="Browse", command=output)
    output_dir=output_entry.get()

    top_frame.pack(side=tk.TOP)
    line.pack(pady=10,fill=tk.X)
    bottom_frame.pack(side=tk.BOTTOM)

    input_path.pack(pady=5,fill=tk.X)
    input_entry.pack(pady=5,fill=tk.X)
    browse1.pack(pady=5)

    output_path.pack(pady=5,fill=tk.X)
    output_entry.pack(pady=5,fill=tk.X)
    browse2.pack(pady=5)

    l1 = tk.Label(master,  text='Select One', width=10 )
    #l1.grid(row=2,column=1)
    choices = [\
        'PCA Analysis',\
        'WL Analysis',\
        'SEA Analysis',\
        'Correlations',\
        'MachineLearning',\
        'ConstructAtomicConfiguration',\
        'ConstructSimulation',\
    ]
    choices2 = ['.dat', '.txt','.pdf', '.png', '.tiff','.jpg']
    variables = tk.StringVar(master)
    variables.set(choices[0])
    global opts
    opts=[choices[0],choices2[0]]
    def my_show(value):
        global opt
        print(value)
        opt=value
        opts[0]=value
    def my_show2(value):
        global opt2
        print(value)
        opt2=value
        opts[1]=value
    opt=choices[0]
    om1 =tk.OptionMenu(master, variables, *choices,command=my_show)
    #b1 = tk.Button(master,  text='Show Value', command=lambda: my_show() )
    om1.pack(pady=5, fill=tk.X)
    #b1.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 


    variables2 = tk.StringVar(master)
    variables2.set(choices2[0])
    opt2=choices2[0]
    om2 =tk.OptionMenu(master, variables2, *choices2,command=my_show2)
    #b2 = tk.Button(master,  text='Show Value', command=lambda: my_show2() )
    #om1.grid(row=2,column=2)
    om2.pack(pady=5, fill=tk.X)
    #b2.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 
    def quitt():
        master.quit()
    submit_button = tk.Button(bottom_frame, text='Submit',command=submit_run)
    clear_button = tk.Button(bottom_frame, text='Clear',command=clear_all)
    end_button = tk.Button(bottom_frame, text='Quit',command=quitt)
    submit_button.pack(pady=10, fill=tk.X)
    clear_button.pack(pady=10, fill=tk.X)
    end_button.pack(pady=10, fill=tk.X)



    master.mainloop()
    print(inp)
    print(outp)
    print(opts[0])
    print(opts[1])
    try:
        return get_path(inp),get_path(outp),opts[0],opts[1]
    except:
        return get_path('.'),get_path('.'),opts[0],opts[1]


def GetOptionsConstrConfig():
    master = tk.Tk()
    master.title('ConstructAtomicConfiguration')
    l1 = tk.Label(master,  text='Select One', width=10 )
    #l1.grid(row=2,column=1)
    choices = [\
        'SEM-EDS Configuration',\
        'Dislocation Configuration'\
    ]
    variables = tk.StringVar(master)
    variables.set(choices[0])
    top_frame = tk.Frame(master)
    bottom_frame = tk.Frame(master)
    line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')
             
    global opts
    opts=[]
    def my_show(value):
        global opt
        print(value)
        opt=value
        opts.append(value)
    def my_show2(value):
        global opt2
        print(value)
        opt2=value
        opts.append(value)
    opt=choices[0]
    om1 =tk.OptionMenu(master, variables, *choices,command=my_show)
    #b1 = tk.Button(master,  text='Show Value', command=lambda: my_show() )
    om1.pack(pady=5, fill=tk.X)
    #b1.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 

    choices2 = ['.dat', '.txt','.pdf', '.png', '.tiff','.jpg']
    variables2 = tk.StringVar(master)
    variables2.set(choices2[0])
    opt2=choices2[0]
    om2 =tk.OptionMenu(master, variables2, *choices2,command=my_show2)
    #b2 = tk.Button(master,  text='Show Value', command=lambda: my_show2() )
    #om1.grid(row=2,column=2)
    om2.pack(pady=5, fill=tk.X)
    #b2.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 
    def quitt():
        master.quit()    
    begin_button = tk.Button(master, text='Done',command=quitt)
    begin_button.pack(pady=5, fill=tk.X)

    master.mainloop()
    return opts[0],opts[1]
