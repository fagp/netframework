import os
import json
import subprocess

used_gpus=list()
touse_gpus=dict()

def list_gpus():
    p = subprocess.Popen(["nvidia-smi","--query-gpu=index,memory.total,memory.used,name", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE)
    (out, _) = p.communicate()
    gpusstr=str(out.decode("utf-8")).split('\n')
    gpus = list()
    for gstr in gpusstr:
        if gstr!='':
            attr=gstr.split(',')
            gpu={'id':int(attr[0]),'name':attr[3][1:],'total':int(attr[1][1:]),'used':int(attr[2][1:])}
            gpus+= [gpu]
            
    return gpus

for gpu in list_gpus():
    touse_gpus[str(gpu['id'])+':'+gpu['name']]=True

def available_gpus():
    gpus=list_gpus()
    available=list()
    global used_gpus
    global touse_gpus

    for gpu in gpus:
        if (touse_gpus[str(gpu['id'])+':'+gpu['name']]) and (gpu['used']/gpu['total'] <0.5 and not gpu['id'] in used_gpus):
            available += [gpu['id']]

    #print(gpus)
    #print(available)
    return available

def list_datasets(projects,pid):
    prj=projects[str(pid)]
    data_path=os.path.join( prj['path'], 'defaults', 'dataconfig_train.json' )
    data=json.load(open(data_path))
    return list(data.keys())

def list_models(projects,pid):
    prj=projects[str(pid)]
    model_path=os.path.join( prj['path'], 'defaults', 'modelconfig.json' )
    model=json.load(open(model_path))
    return list(model.keys())

def list_loss(projects,pid):
    prj=projects[str(pid)]
    loss_path=os.path.join( prj['path'], 'defaults', 'loss_definition.json' )
    loss=json.load(open(loss_path))
    return list(loss.keys())

def load_net(projects,pid):
    prj=projects[str(pid)]
    lfolder=''
    folder_path=prj['path']
    while lfolder=='':
        folder_path,lfolder=os.path.split(folder_path)
    net_path=os.path.join( folder_path, 'out' )

    directories = sorted(os.listdir(net_path))
    models = dict()
    for folder in sorted(directories):
        if not os.path.exists(net_path+'/'+folder+'/model'):
            continue
        list_files=sorted(os.listdir(net_path+'/'+folder+'/model'))
        if list_files:
            models[folder] = list_files
    return models

def load_input(input_path):
    directories = os.listdir(input_path)
    return directories

def dict2str(arguments):
    argsstr=""
    for key,arg in arguments.items():
        if key in ["visdom", "resume", "parallel"]:
            if arg=="True":
                argsstr +=" --"+key
        else:
            if arg!="" and arg!="None":
                if "param" in key:
                    argsstr +=" --"+key+"=\""+arg+"\""
                else:    
                    argsstr +=" --"+key+"="+arg
    return argsstr