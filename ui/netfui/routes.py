from flask import render_template, url_for, request, redirect
from netfui.forms import ProjectForm, TrainForm
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

#models-OK
done_model = model(app.config['paths']['done_path'])
error_model = model(app.config['paths']['error_path'])
started_model=model(app.config['paths']['started_path'])
experiments_model=model(app.config['paths']['queue_path'])
projects_model=model(app.config['paths']['projects_path'])
netfui_path= os.path.dirname( app.config['paths']['queue_path'] )
python_path= app.config['paths']['python_path']

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

def show_progress():
    global process
    while True:
        try:
            for pid,pr in list(process.items()):
                #read std out
                line = pr.stdout.readline()
                if line.decode("utf-8").find('[0/') >=0:
                    started=started_model.list_all()
                    epparse=line.decode("utf-8").split('[')
                    phase=epparse[0][2:-2]
                    cepoch=int(epparse[1][0:-1])
                    if phase=='Valid':
                        cepoch*=-1
                    started[pid]['progress']=cepoch
                    started_model.save(started)

                    socketio.emit('job complete') 
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
    
#begin process for experiment
def begin(expid):
    expid=str(expid) #keys are strings, so cast!
    
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
            if exp['arguments']['experiment']==experiments[expid]['arguments']['experiment']:
                print('Log: Process ',expid,' is running already')
                socketio.emit('job complete') 
                return
                    
        if not expid in list(lock_exp.keys()): #critical region lock
            lock_exp[expid]=1
            print('Log: Starting process ',expid)
            
        else:
            socketio.emit('job complete') 
            return

        if expid!="-1":
            used_gpus += [use_cuda]
            exp=experiments[expid]
            args=exp['arguments']
            args['use_cuda']=str(use_cuda)
            current_proj=projects[exp['pid']]
            exp['progress']='0'
            started=started_model.push_back(exp)
            started_model.save(started)
            experiments=experiments_model.remove(expid)
            experiments_model.save(experiments)

            argsstr=dict2str(args)
            global python_path
            command='exec '+python_path+" -u "+current_proj['exec']+argsstr
            global process
            print(command)
            process[started_model.last_index] = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE ,shell=True, cwd=current_proj['path'])
    
        del lock_exp[expid]


#thread to watch running experiments
def watch_train():
    global process
    global tokill
    global used_gpus #only one job per gpu
    global exp_queue
    while True:
        try:
            started=started_model.list_all()
            for expid, cstarted in list(started.items()):
                if expid in list(tokill.keys()): #if process exist
                    print('Log: Killing process ',tokill[expid].pid)
                    p=tokill[expid]
                    p.kill()
                    del tokill[expid]   

                    error=error_model.push_back(cstarted)
                    error_model.save(error)
                    started=started_model.remove(expid)
                    started_model.save(started)

                    used_gpus.remove(int(cstarted['arguments']['use_cuda']))

                    socketio.emit('job complete')                
                    del process[expid]

                if expid in list(process.keys()): #if process exist
                    pr=process[expid]
                    if pr.poll() is not None: #process ended
                        started=started_model.remove(expid)
                        started_model.save(started)
                        if pr.returncode==0: #job completed 
                            # print('p end well')
                            done=done_model.push_back(cstarted)
                            done_model.save(done)
                        else: #job error
                            # print('p end wrong')
                            error=error_model.push_back(cstarted)
                            error_model.save(error)

                        used_gpus.remove(int(cstarted['arguments']['use_cuda']))

                        socketio.emit('job complete')     
                        print('Log: Training complete ',expid)           
                        del process[expid]    

            if exp_queue:   
                begin(-1)
        except:
            pass
        time.sleep(0.01)

watch_thread = Thread(target=watch_train, daemon=True)
watch_thread.start()

progress_thread = Thread(target=show_progress, daemon=True)
progress_thread.start()

#render list of all experiments-OK
@app.route("/")
@app.route("/home")
def home():
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
        print(started[pid]['progress'])

    done=done_model.list_all()
    global exp_queue
    return render_template('home.html', jobs=jobs, projects=projects, started=started, done=done, error=error,exp_queue=exp_queue)

#render project-OK
@app.route("/project", methods=['GET', 'POST'])
def project():
    projects=projects_model.list_all()
    form=ProjectForm()

    if request.method == 'POST':
        newproj=dict()
        newproj['name']=form.name.data
        newproj['path']=form.path.data
        newproj['exec']=form.exe.data
        projects=projects_model.push_back(newproj)
        projects_model.save(projects)

    return render_template('project.html', title='Projects', projects=projects, form=form)

#remove project on click-OK
@app.route("/project/remove/<int:pid>")
def rmproject(pid):
    projects=projects_model.remove(pid)
    projects_model.save(projects)

    return redirect(url_for('project'))

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
        form.epochs.data=int(job['arguments']['epochs'])
        form.show_rate.data=int(job['arguments']['show_rate'])
        form.print_rate.data=int(job['arguments']['print_rate'])
        form.save_rate.data=int(job['arguments']['save_rate'])
        form.visdom.data=bool(job['arguments']['visdom']=='True')
        form.resume.data=bool(job['arguments']['resume']=='True')
    else:
        form.optimizer.process_data( "Adam" ) #default Adam :)
        form.train_worker.data=0
        form.test_worker.data=0
        form.batch_size.data=1
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
    
#remove queued-OK
@app.route("/experiment/queue/remove/<int:pid>")
def rmqueued(pid):
    experiments=experiments_model.remove(pid)
    experiments_model.save(experiments)

    return redirect(url_for('home'))

#remove done-OK
@app.route("/experiment/done/remove/<int:pid>")
def rmdone(pid):
    done=done_model.remove(pid)
    done_model.save(done)

    return redirect(url_for('home'))

#remove erro-OK
@app.route("/experiment/error/remove/<int:pid>")
def rmerror(pid):
    error=error_model.remove(pid)
    error_model.save(error)

    return redirect(url_for('home'))

#kill started process-OK
@app.route("/experiment/kill/<int:pid>")
def killstarted(pid):
    pid=str(pid)
    global tokill
    tokill[pid]=process[pid]
    
    global exp_queue
    exp_queue=True

    return redirect(url_for('toogle_queue'))

#enqueue failed job-OK
@app.route("/experiment/enqueue/<int:pid>")
def enqueue(pid):
    experiments=experiments_model.push_back(error_model[pid])
    experiments_model.save(experiments)
    error=error_model.remove(pid)
    error_model.save(error)
    
    return redirect(url_for('start'))

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

@app.route("/experiment/down/<int:pid>")
def down(pid):
    pid=str(pid)
    experiments=experiments_model.list_all()
    ids=list(experiments.keys())
    current_ind=ids.index(pid)
    if current_ind<len(ids)-1:
        before_pid=ids[current_ind+1]
        current_experiment=experiments[pid]
        experiments[pid]=experiments[before_pid]
        experiments[before_pid]=current_experiment
        experiments_model.save(experiments)
    
    return redirect(url_for('home'))

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
    
    return render_template('details.html', title='Details',projects=projects,job=job,gpu=gpudict)

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

