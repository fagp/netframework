from flask import render_template, url_for, request, redirect
from netfui.forms import ProjectForm, TrainForm, TestForm, MetricForm
from netfui import app, socketio
from netfui.model_json import model
import os, signal
import sys
import getpass
import json
import datetime
import subprocess
from threading import Thread
from netfui.helpers import *
import time
from queue import Queue, Empty
import select
import numpy as np

#models-OK
done_model = model(os.path.join(app.config['root_path'], app.config['paths']['done_path'] ))
error_model = model(os.path.join(app.config['root_path'], app.config['paths']['error_path']))
started_model=model(os.path.join(app.config['root_path'], app.config['paths']['started_path']))
experiments_model=model(os.path.join(app.config['root_path'], app.config['paths']['queue_path']))
projects_model=model(os.path.join(app.config['root_path'], app.config['paths']['projects_path']))
netfui_path= os.path.dirname(os.path.join(app.config['root_path'],  app.config['paths']['queue_path'] ))
python_path= app.config['py_path']#app.config['paths']['python_path']

#init: if there is any experiment in started then push back to queue-OK
started=started_model.list_all() 
for expid,cstarted in list(started.items()):
    started=started_model.remove(expid)
    started_model.save(started)     
    experiments=experiments_model.push_back(cstarted)
    experiments_model.save(experiments)

#flag to automatically consume from queue-OK
exp_queue=False

process=dict() #used to watch running processes
tokill=dict()
lock_exp=dict()
progress_thread=dict()
error_thread=dict()

def show_error(pid,pr):
    print('Log: Error progress track')
    global process
    time.sleep(1)
    while pid in list(process.keys()):
        try:
            #read std err and add to log
            errline = pr.stderr.readline()
            if errline.decode("utf-8")!='':
                started=started_model.list_all()
                epparse=errline.decode("utf-8")
                if not ('log' in list(started[pid].keys())):
                    started[pid]['log']='Errors:\n'
                now=datetime.datetime.now()
                started[pid]['log']+= str(now.month)+'/'+str(now.day)+'/'+str(now.year)+': '+epparse
                started_model.save(started)

        except:
            pass
        time.sleep(0.01)
    try:
        del error_thread[pid]
    except:
        pass
    print('Log: Ending error progress track')
    socketio.emit('job complete')  

def show_progress(pid,pr):
    print('Log: Train progress track')
    global process
    time.sleep(1)
    while pid in list(process.keys()):
        try:
            #read std out
            line = pr.stdout.readline()
            if line.decode("utf-8").find('[1/') >=0:
                started=started_model.list_all()
                epparse=line.decode("utf-8").split('[')
                phase=epparse[0][2:-2]
                cepoch=int(epparse[1][0:-1])
                if phase=='Valid':
                    cepoch*=-1
                started[pid]['progress']=cepoch
                started_model.save(started)
                socketio.emit('job complete') 

        except:
            pass
        time.sleep(0.01)

    try:
        del progress_thread[pid]
    except:
        pass
    
def show_metric(pid,pr):
    print('Log: Metric progress track')
    global process
    time.sleep(1)
    while pid in list(process.keys()):
        try:
            #read std out
            line = pr.stdout.readline()
            if line.decode("utf-8").find(':') >=0:
                started=started_model.list_all()
                if line.decode("utf-8")=='end':
                    break
                epparse=line.decode("utf-8").split(':')
                metric_name=epparse[0]
                val=float(epparse[1])

                if metric_name not in list(started[pid]['results'].keys()):
                    started[pid]['results'][metric_name] =[]

                started[pid]['results'][metric_name] += [val]
                started_model.save(started)
                socketio.emit('job complete') 

        except:
            pass
        time.sleep(0.01)

    try:
        del progress_thread[pid]
    except:
        pass

#begin process for experiment
def begin(expid):
    expid=str(expid) #keys are strings, so cast!
    
    try:
        experiments=experiments_model.list_all()
        projects=projects_model.list_all()
        agpus=available_gpus()
        started=started_model.list_all()
        global used_gpus
        if len(agpus)>0 and len(list(experiments.keys()))>0:
            if expid=="-1": #select first experiment in queue
                for eid, exp in list(experiments.items()):
                    #if gpu is available or first available (-1)
                    if bool(exp['available']=='True') and ( (int(exp['arguments']['use_cuda']) in agpus) or (int(exp['arguments']['use_cuda'])==-1) ):
                        expid=eid
                        if int(exp['arguments']['use_cuda'])==-1:
                            use_cuda=(agpus[0])
                        else:
                            use_cuda=int(exp['arguments']['use_cuda'])
                        break

            for eid, exp in list(started.items()):
                if expid!="-1" and exp['arguments']['experiment']==experiments[expid]['arguments']['experiment']:
                    print('Log: Process ',exp['arguments']['experiment'],' is running already')
                    socketio.emit('job complete') 
                    return

            if not expid in list(lock_exp.keys()): #critical region lock
                lock_exp[expid]=1
            else:
                socketio.emit('job complete') 
                return

            if expid!="-1":
                used_gpus += [use_cuda]
                exp=experiments[expid]
                now=datetime.datetime.now(); exp['sdate']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
                print('Log: Starting process ',exp['arguments']['experiment'])
                args=exp['arguments']
                if ('test' in list(exp.keys()) and exp['test']=='True'):#if testing
                    exp['log']='Errors:\n'
                    args['epochs']=len(args['inputs'])
                elif args['resume']=='False': #if train and resume is false
                    exp['log']='Errors:\n'

                prev_gpu=args['use_cuda']
                args['use_cuda']=str(use_cuda)
                current_proj=projects[exp['pid']]
                exp['progress']='0'

                if ('test' in list(exp.keys()) and exp['test']=='True'):
                    if (exp['metric']=='True'):
                        argsstr,_=dict2str_metric(args)
                        prexec=current_proj['metric_exec']
                        prpath=current_proj['metric_path']
                    else:
                        key1,key2=args['model'].split('/')
                        nets=load_net(projects,exp['pid'],key1)
                        argsstr=dict2str_test(args,nets)
                        prexec=current_proj['test_exec']
                        prpath=current_proj['test_path']

                else:
                    argsstr=dict2str(args)
                    prexec=current_proj['exec']
                    prpath=current_proj['path']

                exp['used_gpu']=args['use_cuda']
                args['use_cuda']=prev_gpu

                started=started_model.push_back(exp)
                started_model.save(started)
                experiments=experiments_model.remove(expid)
                experiments_model.save(experiments)

                global python_path

                command='exec '+python_path+" -u "+prexec+argsstr
                global process
                print(command)
                slindex=started_model.last_index
                process[slindex] = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE ,shell=True, cwd=prpath)

                if ('test' not in list(exp.keys()) or exp['test']=='False'):
                    progress_thread[slindex] = Thread(target=show_progress,args=(slindex,process[slindex]), daemon=True)
                    progress_thread[slindex].start()

                error_thread[slindex] = Thread(target=show_error,args=(slindex,process[slindex]), daemon=True)
                error_thread[slindex].start()

                if (exp['metric']=='True'):
                    progress_thread[slindex] = Thread(target=show_metric,args=(slindex,process[slindex]), daemon=True)
                    progress_thread[slindex].start()



            del lock_exp[expid]
    except:
        pass


#thread to watch running experiments
def watch_train():
    print('Log: Init watch process')
    global process
    global tokill
    global used_gpus #only one job per gpu
    global exp_queue
    while True:
        try:
            started=started_model.list_all()
            for expid, cstarted in list(started.items()):
                if expid in list(tokill.keys()): #if process exist
                    print('Log: Killing process ',cstarted['arguments']['experiment'])
                    p=tokill[expid]
                    p.kill()
                    del tokill[expid]   

                    error=error_model.push_back(cstarted)
                    error_model.save(error)
                    started=started_model.remove(expid)
                    started_model.save(started)

                    used_gpus.remove(int(cstarted['used_gpu']))

                    socketio.emit('job complete')                
                    del process[expid]

                if expid in list(process.keys()): #if process exist
                    pr=process[expid]
                    if pr.poll() is not None: #process ended
                        finished=False
                        started=started_model.remove(expid)
                        started_model.save(started)
                        if pr.returncode==0: #job completed 
                            # print('p end well')
                            if ('test' in list(cstarted.keys()) and cstarted['test']=='True' and ( ( cstarted['metric']=='True' and len(cstarted['arguments']['inputs'])>0) or (cstarted['metric']=='False' and len(cstarted['arguments']['inputs'])>1)  )      ): #if the process is test type and there are more inputs in the queue
                                if cstarted['metric']=='False':
                                    last_input=cstarted['arguments']['inputs'].pop(0)
                                    cstarted['arguments']['pinputs']+=[last_input]
                                args=cstarted['arguments']
                                prev_gpu=args['use_cuda']
                                args['use_cuda']=cstarted['used_gpu']
                                projects=projects_model.list_all()
                                current_proj=projects[cstarted['pid']]

                                if cstarted['metric']=='True':
                                    argsstr,out=dict2str_metric(args)
                                    last_input=cstarted['arguments']['outputs'].pop(0)
                                    cstarted['arguments']['poutputs']+=[last_input]
                                    cstarted['arguments']['inputs'].remove(out[0])
                                    cstarted['arguments']['pinputs']+=out

                                    prexec=current_proj['metric_exec']
                                    prpath=current_proj['metric_path']
                                else:
                                    key1,key2=args['model'].split('/')
                                    nets=load_net(projects,cstarted['pid'],key1)
                                    argsstr=dict2str_test(args,nets)
                                    prexec=current_proj['test_exec']
                                    prpath=current_proj['test_path']
                                cstarted['used_gpu']=args['use_cuda']
                                args['use_cuda']=prev_gpu
                                global python_path
            
                                command='exec '+python_path+" -u "+prexec+argsstr
                                print(command)
                                del process[expid] 
                                cstarted['progress']= len(cstarted['arguments']['pinputs'])
                                started=started_model.push_back(cstarted)
                                started_model.save(started)
                                slindex=started_model.last_index
                                process[slindex] = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE ,shell=True, cwd=prpath)
                                error_thread[slindex] = Thread(target=show_error,args=(slindex,process[slindex]), daemon=True)
                                error_thread[slindex].start()

                                if cstarted['metric']=='True':
                                    progress_thread[slindex] = Thread(target=show_metric,args=(slindex,process[slindex]), daemon=True)
                                    progress_thread[slindex].start()

                                socketio.emit('job complete')   

                            else: #if training ends or all test inputs in queue were processed
                                if cstarted['test']=='True' and cstarted['metric']=='False':
                                    last_input=cstarted['arguments']['inputs'].pop(0)
                                    cstarted['arguments']['pinputs']+=[last_input]
                                now=datetime.datetime.now(); cstarted['ddate']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
                                done=done_model.push_back(cstarted)
                                done_model.save(done)
                                print('Log: Process complete ',cstarted['arguments']['experiment'])
                                finished=True
                            
                            
                        else: #job error
                            # print('p end wrong')
                            error=error_model.push_back(cstarted)
                            error_model.save(error)
                            print('Log: Error in process ',cstarted['arguments']['experiment'])
                            finished=True

                        if finished:
                            used_gpus.remove(int(cstarted['used_gpu']))  
                            del process[expid] 
                        socketio.emit('job complete')  
                                   
            # if exp_queue:   
            #     begin(-1)
        except:
            pass
        time.sleep(0.1)

watch_thread = Thread(target=watch_train, daemon=True)
watch_thread.start()

#render list of all experiments-OK
@app.route("/")
@app.route("/home")
def home():

    gpus=list_gpus()
    if not gpus:
        return render_template('error.html', errortitle='No GPU found!',error="It seems that you do not have a Nvidia GPU. \n Only GPU schedule is allowed... for now" )

    error=error_model.list_all()
    jobs=experiments_model.list_all()
    projects=projects_model.list_all()
    started=started_model.list_all()
    for pid,_ in list(started.items()):
        v=float(started[pid]['progress'])
        started[pid]['valid']=False
        if v<0:
            started[pid]['valid']=True
            v*=-1
        started[pid]['progress']=100*v/float(started[pid]['arguments']['epochs'])
        #print(started[pid]['progress'])

    done=done_model.list_all()
    global exp_queue
    now=datetime.datetime.now(); ctime=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
    return render_template('home.html', jobs=jobs, projects=projects, started=started, done=done, error=error,exp_queue=exp_queue,tuse_gpu=touse_gpus, time=ctime)

#render project-OK
@app.route("/project", methods=['GET', 'POST'])
@app.route("/project/<int:pid>", methods=['GET', 'POST'])
def project(pid=-1):
    pid=str(pid)
    projects=projects_model.list_all()
    form=ProjectForm()

    if request.method == 'POST':
        if pid!='-1':
            newproj=projects[pid]
        else:
            newproj=dict()

        newproj['name']=form.name.data
        newproj['path']=form.path.data
        newproj['exec']=form.exe.data
        newproj['test_path']=form.test_path.data
        newproj['test_exec']=form.test_exe.data
        newproj['metric_path']=form.metric_path.data
        newproj['metric_exec']=form.metric_exe.data
        
        if pid!='-1':
            projects=projects_model.insert(pid,newproj)
            projects_model.save(projects)
        else:
            projects=projects_model.push_back(newproj)
            projects_model.save(projects)
        
        return redirect(url_for('project'))

    if pid!='-1':
        try:
            newproj=projects[pid]
            form.name.data = newproj['name']
            form.path.data = newproj['path']      
            form.exe.data = newproj['exec']    
            form.test_path.data = newproj['test_path']
            form.test_exe.data = newproj['test_exec'] 
            form.metric_path.data = newproj['metric_path']
            form.metric_exe.data = newproj['metric_exec']
        except:
            pass

    return render_template('project.html', title='Projects', projects=projects, form=form)

#remove project on click-OK
@app.route("/project/remove/<int:pid>")
def rmproject(pid):
    projects=projects_model.remove(pid)
    projects_model.save(projects)

    return redirect(url_for('project'))

#add experiment-OK
@app.route("/metric", methods=['GET', 'POST'])
@app.route("/metric/<int:pid>", methods=['GET', 'POST'])
@app.route("/metric/clone_queue/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_queue/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_run/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_run/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_error/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_error/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_done/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/clone_done/<int:expid>", methods=['GET', 'POST'])
def metric(pid=-1,expid=-1):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=MetricForm()
    
    first_project=False
    if pid==-1: #select first project
        pid=list(projects.keys())[0]
        first_project=True

    rule = request.url_rule
    if 'done' in rule.rule:
        job=done_model[expid]
    elif 'error' in rule.rule:
        job=error_model[expid]
    elif 'queue' in rule.rule:
        job=experiments_model[expid]
    elif 'run' in rule.rule:
        job=started_model[expid]

    if ('clone' in rule.rule) and (first_project):
        pid=job['pid']

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
        exp['available']='True'
        exp['progress']='0'
        exp['test']='True'
        exp['metric']='True'
        exp['results']=dict()
        now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
        args=dict()
        args['experiment']      =str(form.experiment.data)
        args['pathinputs']      =str(form.inputs.data)
        args['pathoutputs']     =str(form.outputs.data)
        args['outputs']          =load_input(str(form.outputs.data))
        args['outputsarg']      =str(form.outputsarg.data)
        args['inputs']          =load_input(str(form.inputs.data))
        args['inputsarg']       =str(form.inputsarg.data)
        args['pinputs']         =[]
        args['poutputs']         =[]
        args['otherarg']        =str(form.otherarg.data)
        args['use_cuda']        =str(form.use_cuda.data)
        exp['arguments']=args

        if len(args['outputs'] )!=len(args['inputs']):
            exp['available']='False'

        jobs=experiments_model.push_back(exp)
        experiments_model.save(jobs)
        return redirect(url_for('start'))

    for projid,proj in projects.items(): #populate project select
        form.project.choices +=[( url_for('metric', pid=projid), proj['name'])]
        
    form.project.process_data( url_for('metric', pid=pid) )

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]
    
    if 'clone' in rule.rule:
        form.inputs.data=(job['arguments']['pathinputs'])
        form.inputsarg.data=(job['arguments']['inputsarg'])
        form.otherarg.data=(job['arguments']['otherarg'])
        form.outputs.data=(job['arguments']['pathoutputs'])
        form.outputsarg.data=(job['arguments']['outputsarg'])
        

    return render_template('metric.html', title='Add Metric', form=form, pid=pid)

#add experiment-OK
@app.route("/test", methods=['GET', 'POST'])
@app.route("/test/<int:pid>", methods=['GET', 'POST'])
@app.route("/test/clone_queue/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_queue/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_run/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_run/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_error/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_error/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_done/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/clone_done/<int:expid>", methods=['GET', 'POST'])
def test(pid=-1,expid=-1):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=TestForm()
    
    first_project=False
    if pid==-1: #select first project
        pid=list(projects.keys())[0]
        first_project=True

    rule = request.url_rule
    if 'done' in rule.rule:
        job=done_model[expid]
    elif 'error' in rule.rule:
        job=error_model[expid]
    elif 'queue' in rule.rule:
        job=experiments_model[expid]
    elif 'run' in rule.rule:
        job=started_model[expid]

    if ('clone' in rule.rule) and (first_project):
        pid=job['pid']

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
        exp['available']='True'
        exp['progress']='0'
        exp['test']='True'
        exp['metric']='False'
        now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
        args=dict()
        args['experiment']      =str(form.experiment.data)
        args['model']           =str(form.model.data)
        args['modelarg']        =str(form.modelarg.data)
        args['pathinputs']      =str(form.inputs.data)
        args['outputs']         =str(form.outputs.data)
        args['outputsarg']      =str(form.outputsarg.data)
        args['inputs']          =load_input(str(form.inputs.data))
        args['inputsarg']       =str(form.inputsarg.data)
        args['pinputs']         =[]
        args['otherarg']        =str(form.otherarg.data)
        args['use_cuda']        =str(form.use_cuda.data)
        exp['arguments']=args

        jobs=experiments_model.push_back(exp)
        experiments_model.save(jobs)
        return redirect(url_for('start'))

    for projid,proj in projects.items(): #populate project select
        form.project.choices +=[( url_for('test', pid=projid), proj['name'])]
        
    form.project.process_data( url_for('test', pid=pid) )
    
    nets=load_net(projects,pid)
    for netname,net in nets.items(): #populate model select according to selected project
        choices =((netname+'/all', netname+':all'),)
        for netepoch,_ in net.items():
            tup=(netname+'/'+netepoch, netname+':'+ netepoch.replace('model.t7',''))
            choices += (tup,)
        tup=(netname, choices)
        form.model.choices += (tup,)

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]
    
    if 'clone' in rule.rule:
        if first_project:
            form.model.process_data( (job['arguments']['model']) )
        form.modelarg.process_data( (job['arguments']['modelarg']) )
        form.inputs.data=(job['arguments']['pathinputs'])
        form.inputsarg.data=(job['arguments']['inputsarg'])
        form.otherarg.data=(job['arguments']['otherarg'])
        form.outputs.data=(job['arguments']['outputs'])
        form.outputsarg.data=(job['arguments']['outputsarg'])
        

    return render_template('test.html', title='Add Test', form=form, pid=pid)

#add experiment-OK
@app.route("/experiment", methods=['GET', 'POST'])
@app.route("/experiment/<int:pid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_queue/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_queue/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_run/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_run/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_error/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_error/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_done/<int:pid>/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/clone_done/<int:expid>", methods=['GET', 'POST'])
def experiment(pid=-1,expid=-1):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=TrainForm()
    
    first_project=False
    if pid==-1: #select first project
        pid=list(projects.keys())[0]
        first_project=True

    rule = request.url_rule
    if 'done' in rule.rule:
        job=done_model[expid]
    elif 'error' in rule.rule:
        job=error_model[expid]
    elif 'queue' in rule.rule:
        job=experiments_model[expid]
    elif 'run' in rule.rule:
        job=started_model[expid]

    if ('clone' in rule.rule) and (first_project):
        pid=job['pid']

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
        exp['available']='True'
        exp['progress']='0'
        exp['test']='False'
        exp['metric']='False'
        now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
        args=dict()
        args['experiment']      =str(form.experiment.data)
        args['dataset']         =str(form.dataset.data)
        args['datasetparam']    =str(form.datasetparam.data)
        args['model']           =str(form.model.data)
        args['modelparam']      =str(form.modelparam.data)
        args['optimizer']       =str(form.optimizer.data)
        args['optimizerparam']  =str(form.optimizerparam.data)
        args['lrschedule']      =str(form.lrschedule.data)
        args['loss']            =str(form.loss.data)
        args['lossparam']       =str(form.lossparam.data)
        args['epochs']          =str(form.epochs.data)
        args['batch_size']      =str(form.batch_size.data)
        args['batch_acc']      =str(form.batch_acc.data)
        args['visdom']          =str(form.visdom.data)
        args['show_rate']       =str(form.show_rate.data)
        args['print_rate']      =str(form.print_rate.data)
        args['save_rate']       =str(form.save_rate.data)
        args['use_cuda']        =str(form.use_cuda.data)
        args['parallel']        =str(form.parallel.data)
        args['train_worker']    =str(form.train_worker.data)
        args['test_worker']     =str(form.test_worker.data)
        args['resume']          =str(form.resume.data)
        exp['arguments']=args

        jobs=experiments_model.push_back(exp)
        experiments_model.save(jobs)
        return redirect(url_for('start'))

    for projid,proj in projects.items(): #populate project select
        form.project.choices +=[( url_for('experiment', pid=projid), proj['name'])]
        
    form.project.process_data( url_for('experiment', pid=pid) )
    
    datasets=list_datasets(projects,pid)
    for data in datasets: #populate dataset select according to selected project
        form.dataset.choices +=[(data, data)]
    
    models=list_models(projects,pid)
    for model in models: #populate model select according to selected project
        form.model.choices +=[(model, model)]

    losses=list_loss(projects,pid)
    for loss in losses: #populate loss select according to selected project
        form.loss.choices +=[(loss, loss)]

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]
    

    if 'clone' in rule.rule:
        if first_project:
            form.optimizer.process_data( (job['arguments']['optimizer']) )
            form.loss.process_data( (job['arguments']['loss']) )
            form.model.process_data( (job['arguments']['model']) )
            form.dataset.process_data( (job['arguments']['dataset']) )
        form.lrschedule.process_data( (job['arguments']['lrschedule']) )
        form.datasetparam.data=(job['arguments']['datasetparam'])
        form.modelparam.data=(job['arguments']['modelparam'])
        form.lossparam.data=(job['arguments']['lossparam'])
        form.optimizerparam.data=(job['arguments']['optimizerparam'])
        form.train_worker.data=int(job['arguments']['train_worker'])
        form.test_worker.data=int(job['arguments']['test_worker'])
        form.batch_size.data=int(job['arguments']['batch_size'])
        if 'batch_acc'in job['arguments'].keys():
            form.batch_acc.data=int(job['arguments']['batch_acc'])
        else:
            form.batch_acc.data=1
        form.epochs.data=int(job['arguments']['epochs'])
        form.show_rate.data=int(job['arguments']['show_rate'])
        form.print_rate.data=int(job['arguments']['print_rate'])
        form.save_rate.data=int(job['arguments']['save_rate'])
        form.visdom.data=bool(job['arguments']['visdom']=='True')
        form.resume.data=bool(job['arguments']['resume']=='True')
    else:
        form.optimizer.process_data( "Adam" ) #default optimizer Adam
        form.train_worker.data=0
        form.test_worker.data=0
        form.batch_size.data=1
        form.batch_acc.data=1
        form.epochs.data=1000
        form.show_rate.data=5
        form.print_rate.data=5
        form.save_rate.data=20
        form.visdom.data=True

    return render_template('experiment.html', title='Add Experiment', form=form, pid=pid)

#toogle queue on/off-OK
@app.route("/experiment/toogle_queue")
def toogle_queue():
    global exp_queue
    exp_queue=not exp_queue
    return redirect(url_for('start'))

#toogle gpu on/off-OK
@app.route("/experiment/toogle_gpu/<gpuname>")
def toogle_gpu(gpuname):
    global touse_gpus
    touse_gpus[gpuname]=not touse_gpus[gpuname]
    return redirect(url_for('start'))

#start experiments in queue-OK
@app.route("/experiment/start")
def start():
    global exp_queue
    if exp_queue:
        return redirect(url_for('start_experiment'))
    return redirect(url_for('home'))

#start custom experiment-OK
@app.route("/experiment/start_first")
@app.route("/experiment/start/<int:expid>")
def start_experiment(expid=-1):
    begin(expid)
    return redirect(url_for('home'))
    
#remove experiments-OK
@app.route("/experiment/remove_queue/<int:pid>")
@app.route("/experiment/remove_done/<int:pid>")
@app.route("/experiment/remove_error/<int:pid>")
def rm(pid):

    rule = request.url_rule
    if 'done' in rule.rule:
        job=done_model.remove(pid)
        done_model.save(job)
    elif 'error' in rule.rule:
        job=error_model.remove(pid)
        error_model.save(job)
    elif 'queue' in rule.rule:
        job=experiments_model.remove(pid)
        experiments_model.save(job)

    return redirect(url_for('home'))

#kill started process-OK
@app.route("/experiment/kill/<int:pid>")
def killstarted(pid):
    pid=str(pid)
    
    global tokill
    tokill[pid]=process[pid] #add process in kill list
    
    global exp_queue
    exp_queue=False #stop queue processing

    return redirect(url_for('start'))

#enqueue failed job-OK
@app.route("/experiment/enqueue/<int:pid>")
def enqueue(pid):
    experiments=experiments_model.push_back(error_model[pid])
    experiments_model.save(experiments)
    error=error_model.remove(pid)
    error_model.save(error)
    
    return redirect(url_for('start'))

#up experiment in queue-OK
@app.route("/experiment/up/<int:pid>")
def up(pid):
    pid=str(pid)
    experiments=experiments_model.list_all()
    ids=list(experiments.keys())
    current_ind=ids.index(pid)
    if current_ind>0:
        before_pid=ids[current_ind-1]
        current_experiment=experiments[pid]
        experiments[pid]=experiments[before_pid]
        experiments[before_pid]=current_experiment
        experiments_model.save(experiments)
    
    return redirect(url_for('home'))

#down experiment in queue-OK
@app.route("/experiment/down/<int:pid>")
def down(pid):
    pid=str(pid)
    experiments=experiments_model.list_all()
    ids=list(experiments.keys())
    current_ind=ids.index(pid)
    if current_ind<len(ids)-1:
        after_pid=ids[current_ind+1]
        current_experiment=experiments[pid]
        experiments[pid]=experiments[after_pid]
        experiments[after_pid]=current_experiment
        experiments_model.save(experiments)
    
    return redirect(url_for('home'))

#details of the experiments-OK
@app.route("/experiment/details_done/<int:pid>")
@app.route("/experiment/details_error/<int:pid>")
@app.route("/experiment/details_run/<int:pid>")
@app.route("/experiment/details_queue/<int:pid>")
def details(pid):
    rule = request.url_rule
    if 'done' in rule.rule:
        job=done_model[pid]
    elif 'error' in rule.rule:
        job=error_model[pid]
    elif 'queue' in rule.rule:
        job=experiments_model[pid]
    elif 'run' in rule.rule:
        job=started_model[pid]

    projects=projects_model.list_all()
    gpus=list_gpus()
    gpudict=dict()
    for gpu in gpus:
        gpudict[str(gpu['id'])]=gpu['name']
    gpudict['-1']='First Available'

    for key in ['inputs', 'pinputs', 'outputs', 'poutputs']:
        if key in list(job['arguments'].keys()) and isinstance(job['arguments'][key],list):
            l=min(11,len(job['arguments'][key]))
            for i in range(l-1):
                _,filename=os.path.split(job['arguments'][key][i])
                job['arguments'][key][i]=filename
                if i >11:
                    break

    return render_template('details.html', title='Details',projects=projects,job=job,gpu=gpudict)

@app.route("/metric/result/<int:pid>")
def detailsmetrics(pid):
    rule = request.url_rule
    job=done_model[pid]
    
    projects=projects_model.list_all()

    job['arguments']['index']=[]
    for i in range(len(job['arguments']['pinputs'])):
        _,filename=os.path.split(job['arguments']['pinputs'][i])
        job['arguments']['pinputs'][i]=filename
        _,filename=os.path.split(job['arguments']['poutputs'][i])
        job['arguments']['poutputs'][i]=filename
        job['arguments']['index']+=[i]

    means=dict()
    for metric, vals in job['results'].items():
        means[metric]= np.mean(np.array(vals[1:]))
        
    return render_template('details_metrics.html', title='Details',projects=projects,job=job, means=means)


#update experiments in queue and error-OK
@app.route("/metric/update_queue/<int:expid>", methods=['GET', 'POST'])
@app.route("/metric/update_error/<int:expid>", methods=['GET', 'POST'])
def update_metric(expid):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=MetricForm()
    rule = request.url_rule
    
    if 'error' in rule.rule:
        job=error_model[expid]
        error_model.disable(expid)
    elif 'queue' in rule.rule:
        job=experiments_model[expid]
        experiments_model.disable(expid)
    pid=job['pid']

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
        exp['available']='True'
        exp['progress']='0'
        exp['test']='True'
        exp['metric']='True'
        exp['results']=dict()
        now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
        args=dict()
        args['experiment']      =str(form.experiment.data)
        args['pathoutputs']      =str(form.outputs.data)
        args['outputs']         =load_input(str(form.outputs.data))
        args['outputsarg']      =str(form.outputsarg.data)
        args['pathinputs']      =str(form.inputs.data)
        args['inputs']          =load_input(str(form.inputs.data))
        args['inputsarg']       =str(form.inputsarg.data)
        args['pinputs']         =[]
        args['poutputs']         =[]
        args['otherarg']        =str(form.otherarg.data)
        args['use_cuda']        =str(form.use_cuda.data)
        exp['arguments']=args

        if len(args['outputs'] )!=len(args['inputs']):
            exp['available']='False'

        if 'error' in rule.rule:
            jobs=error_model.list_all()
            jobs[expid]=exp
            error_model.save(jobs)
        elif 'queue' in rule.rule:
            jobs=experiments_model.list_all()
            jobs[expid]=exp
            experiments_model.save(jobs)

        return redirect(url_for('start'))

    for projid,proj in projects.items(): #populate project select
        form.project.choices +=[( url_for('metric', pid=projid), proj['name'])]
        
    form.project.process_data( url_for('metric', pid=pid) )

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]
    
    form.experiment.data=(job['arguments']['experiment'])
    form.inputs.data=(job['arguments']['pathinputs'])
    form.inputsarg.data=(job['arguments']['inputsarg'])
    form.outputs.data=(job['arguments']['pathoutputs'])
    form.outputsarg.data=(job['arguments']['outputsarg'])
    form.otherarg.data=(job['arguments']['otherarg'])
    form.use_cuda.process_data( int(job['arguments']['use_cuda']) )
    
    return render_template('metric.html', title='Update Metric', form=form, pid=pid)

@app.route("/test/metric/<int:expid>")
def create_metric(expid):

    job=done_model[expid]

    exp=dict()
    exp['user']=getpass.getuser()
    exp['pid']=job['pid']
    exp['available']='False'
    exp['progress']='0'
    exp['test']='True'
    exp['metric']='True'
    now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
    args=dict()
    args['experiment']      =job['arguments']['experiment']+'_metric'
    args['pathoutputs']     =''
    args['outputs']         =''
    args['outputsarg']      =''
    args['pathinputs']      =os.path.join(job['arguments']['outputs'],job['arguments']['experiment'])
    args['inputs']          =''
    args['inputsarg']       =''
    args['pinputs']         =[]
    args['poutputs']         =[]
    args['otherarg']        =''
    args['use_cuda']        =str(-1)
    exp['arguments']=args

    jobs=experiments_model.push_back(exp)
    experiments_model.save(jobs)

    return redirect('/metric/update_queue/'+str(experiments_model.last_index))

#update experiments in queue and error-OK
@app.route("/test/update_queue/<int:expid>", methods=['GET', 'POST'])
@app.route("/test/update_error/<int:expid>", methods=['GET', 'POST'])
def update_test(expid):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=TestForm()
    rule = request.url_rule
    
    if 'error' in rule.rule:
        job=error_model[expid]
        error_model.disable(expid)
    elif 'queue' in rule.rule:
        job=experiments_model[expid]
        experiments_model.disable(expid)
    pid=job['pid']

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
        exp['available']='True'
        exp['progress']='0'
        exp['test']='True'
        exp['metric']='False'
        now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
        args=dict()
        args['experiment']      =str(form.experiment.data)
        args['model']           =str(form.model.data)
        args['modelarg']        =str(form.modelarg.data)
        args['outputs']         =str(form.outputs.data)
        args['outputsarg']      =str(form.outputsarg.data)
        args['pathinputs']      =str(form.inputs.data)
        args['inputs']          =load_input(str(form.inputs.data))
        args['inputsarg']       =str(form.inputsarg.data)
        args['pinputs']         =[]
        args['otherarg']        =str(form.otherarg.data)
        args['use_cuda']        =str(form.use_cuda.data)
        exp['arguments']=args

        if 'error' in rule.rule:
            jobs=error_model.list_all()
            jobs[expid]=exp
            error_model.save(jobs)
        elif 'queue' in rule.rule:
            jobs=experiments_model.list_all()
            jobs[expid]=exp
            experiments_model.save(jobs)

        return redirect(url_for('start'))

    for projid,proj in projects.items(): #populate project select
        form.project.choices +=[( url_for('test', pid=projid), proj['name'])]
        
    form.project.process_data( url_for('test', pid=pid) )
    
    nets=load_net(projects,pid)
    for netname,net in nets.items(): #populate model select according to selected project
        choices =((netname+'/all', netname+':all'),)
        for netepoch in net:
            tup=(netname+'/'+netepoch, netname+':'+ netepoch.replace('model.t7',''))
            choices += (tup,)
        tup=(netname, choices)
        form.model.choices += (tup,)

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]
    
    form.model.process_data( (job['arguments']['model']) )
    form.modelarg.process_data( (job['arguments']['modelarg']) )
    form.experiment.data=(job['arguments']['experiment'])
    form.inputs.data=(job['arguments']['pathinputs'])
    form.inputsarg.data=(job['arguments']['inputsarg'])
    form.outputs.data=(job['arguments']['outputs'])
    form.outputsarg.data=(job['arguments']['outputsarg'])
    form.otherarg.data=(job['arguments']['otherarg'])
    form.use_cuda.process_data( int(job['arguments']['use_cuda']) )
    
    return render_template('test.html', title='Update Test', form=form, pid=pid)

@app.route("/experiment/test/<int:expid>")
def create_test(expid):

    job=done_model[expid]

    exp=dict()
    exp['user']=getpass.getuser()
    exp['pid']=job['pid']
    exp['available']='False'
    exp['progress']='0'
    exp['test']='True'
    exp['metric']='False'
    now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)+' '+str(now.hour)+':'+str(now.minute)
    args=dict()
    args['experiment']      =job['arguments']['experiment']+'_test'
    args['model']           =job['arguments']['experiment']+'/lastmodel.t7'
    args['modelarg']        =''
    args['outputs']         =''
    args['outputsarg']      =''
    args['pathinputs']      =''
    args['inputs']          =''
    args['inputsarg']       =''
    args['pinputs']         =[]
    args['otherarg']        =''
    args['use_cuda']        =str(-1)
    exp['arguments']=args

    jobs=experiments_model.push_back(exp)
    experiments_model.save(jobs)

    return redirect('/test/update_queue/'+str(experiments_model.last_index))



#update experiments in queue and error-OK
@app.route("/experiment/update_queue/<int:expid>", methods=['GET', 'POST'])
@app.route("/experiment/update_error/<int:expid>", methods=['GET', 'POST'])
def update(expid):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=TrainForm()
    rule = request.url_rule
    if 'error' in rule.rule:
        job=error_model[expid]
        error_model.disable(expid)
    elif 'queue' in rule.rule:
        job=experiments_model[expid]
        experiments_model.disable(expid)
    pid=job['pid']

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
        exp['available']='True'
        exp['progress']=job['progress']
        exp['test']='False'
        exp['metric']='False'
        now=datetime.datetime.now(); exp['date']=str(now.month)+'/'+str(now.day)+'/'+str(now.year)
        args=dict()
        args['experiment']      =str(form.experiment.data)
        args['dataset']         =str(form.dataset.data)
        args['datasetparam']    =str(form.datasetparam.data)
        args['model']           =str(form.model.data)
        args['modelparam']      =str(form.modelparam.data)
        args['optimizer']       =str(form.optimizer.data)
        args['optimizerparam']  =str(form.optimizerparam.data)
        args['lrschedule']      =str(form.lrschedule.data)
        args['loss']            =str(form.loss.data)
        args['lossparam']       =str(form.lossparam.data)
        args['epochs']          =str(form.epochs.data)
        args['batch_size']      =str(form.batch_size.data)
        args['batch_acc']      =str(form.batch_acc.data)
        args['visdom']          =str(form.visdom.data)
        args['show_rate']       =str(form.show_rate.data)
        args['print_rate']      =str(form.print_rate.data)
        args['save_rate']       =str(form.save_rate.data)
        args['use_cuda']        =str(form.use_cuda.data)
        args['parallel']        =str(form.parallel.data)
        args['train_worker']    =str(form.train_worker.data)
        args['test_worker']     =str(form.test_worker.data)
        args['resume']          =str(form.resume.data)
        exp['arguments']=args

        if 'error' in rule.rule:
            jobs=error_model.list_all()
            jobs[expid]=exp
            error_model.save(jobs)
        elif 'queue' in rule.rule:
            jobs=experiments_model.list_all()
            jobs[expid]=exp
            experiments_model.save(jobs)

        return redirect(url_for('start'))

    for projid,proj in projects.items(): #populate project select
        form.project.choices +=[( url_for('experiment', pid=projid), proj['name'])]
        
    form.project.process_data( url_for('experiment', pid=pid) )
    
    datasets=list_datasets(projects,pid)
    for data in datasets: #populate dataset select according to selected project
        form.dataset.choices +=[(data, data)]
    
    models=list_models(projects,pid)
    for model in models: #populate model select according to selected project
        form.model.choices +=[(model, model)]

    losses=list_loss(projects,pid)
    for loss in losses: #populate loss select according to selected project
        form.loss.choices +=[(loss, loss)]

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]

    form.use_cuda.process_data( int(job['arguments']['use_cuda']) )
    form.optimizer.process_data( (job['arguments']['optimizer']) )
    form.loss.process_data( (job['arguments']['loss']) )
    form.model.process_data( (job['arguments']['model']) )
    form.dataset.process_data( (job['arguments']['dataset']) )
    form.lrschedule.process_data( (job['arguments']['lrschedule']) )
    form.datasetparam.data=(job['arguments']['datasetparam'])
    form.modelparam.data=(job['arguments']['modelparam'])
    form.lossparam.data=(job['arguments']['lossparam'])
    form.experiment.data=(job['arguments']['experiment'])
    form.optimizerparam.data=(job['arguments']['optimizerparam'])
    form.train_worker.data=int(job['arguments']['train_worker'])
    form.test_worker.data=int(job['arguments']['test_worker'])
    form.batch_size.data=int(job['arguments']['batch_size'])
    if 'batch_acc'in job['arguments'].keys():
            form.batch_acc.data=int(job['arguments']['batch_acc'])
    else:
            form.batch_acc.data=1
    form.epochs.data=int(job['arguments']['epochs'])
    form.show_rate.data=int(job['arguments']['show_rate'])
    form.print_rate.data=int(job['arguments']['print_rate'])
    form.save_rate.data=int(job['arguments']['save_rate'])
    form.visdom.data=bool(job['arguments']['visdom']=='True' )
    form.resume.data=bool(job['arguments']['resume']=='True' )
    form.parallel.data=bool(job['arguments']['parallel']=='True' )
    
    return render_template('experiment.html', title='Update Experiment', form=form, pid=pid)

@app.route("/about")
def about():
    return render_template('about.html', title='About')

