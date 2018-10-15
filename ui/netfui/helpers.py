import os
import json
import subprocess

used_gpus=list()

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

def available_gpus():
    gpus=list_gpus()
    available=list()
    global used_gpus
    for gpu in gpus:
        if gpu['used']/gpu['total'] <0.5 and not gpu['id'] in used_gpus:
            available += [gpu['id']]

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