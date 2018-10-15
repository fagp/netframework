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

#models-OK
done_model = model(app.config['paths']['done_path'])
error_model = model(app.config['paths']['error_path'])
started_model=model(app.config['paths']['started_path'])
experiments_model=model(app.config['paths']['queue_path'])
projects_model=model(app.config['paths']['projects_path'])

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
#thread to watch running experiments
def watch_train():
    global process
    while True:
        started=started_model.list_all()
        for expid, cstarted in list(started.items()):
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
                    
                    global used_gpus #only one job per gpu
                    used_gpus.remove(int(cstarted['arguments']['use_cuda']))
                    
                    socketio.emit('job complete')                
                    del process[expid]    
                
        time.sleep(1)

t = Thread(target=watch_train, daemon=True)
t.start()

#render list of all experiments-OK
@app.route("/")
@app.route("/home")
def home():
    error=error_model.list_all()
    jobs=experiments_model.list_all()
    projects=projects_model.list_all()
    started=started_model.list_all()
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
def experiment(pid=-1):
    projects= projects_model.list_all()
    if len(list(projects.keys()))==0: #if there is no project redirect to project page
        return redirect(url_for('project'))

    form=TrainForm()
    if pid==-1: #select first project
        pid=list(projects.keys())[0]

    if request.method == 'POST':
        exp=dict()
        exp['user']=getpass.getuser()
        exp['pid']=str(pid)
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

    form.optimizer.process_data( "Adam" ) #default Adam :)

    gpus=list_gpus()
    form.use_cuda.choices += [(-1,'First Available')]
    for gpu in gpus: #populate gpu select with available gpus
        form.use_cuda.choices +=[(gpu['id'], gpu['name'])]
    
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
    expid=str(expid) #keys are strings, so cast!
    experiments=experiments_model.list_all()
    projects=projects_model.list_all()
    agpus=available_gpus()
    global used_gpus
    if len(agpus)>0 and len(list(experiments.keys()))>0:
        if expid=="-1": #select first experiment in queue
            for eid, exp in list(experiments.items()):
                #if gpu is available or first available (-1)
                if (int(exp['arguments']['use_cuda']) in agpus) or (int(exp['arguments']['use_cuda'])==-1):
                    expid=eid
                    if int(exp['arguments']['use_cuda'])==-1:
                        use_cuda=(agpus[0])
                    else:
                        use_cuda=int(exp['arguments']['use_cuda'])
                    used_gpus += [use_cuda]
                    break

        if expid!="-1":
            exp=experiments[expid]
            args=exp['arguments']
            args['use_cuda']=str(use_cuda)
            current_proj=projects[exp['pid']]
            started=started_model.push_back(exp)
            started_model.save(started)
            experiments=experiments_model.remove(expid)
            experiments_model.save(experiments)

            argsstr=dict2str(args)
            command=os.environ['_']+" "+current_proj['exec']+argsstr
            global process
            print(command)
            process[started_model.last_index] = subprocess.Popen(command, stderr=subprocess.PIPE,stdout=subprocess.PIPE, shell=True,cwd=current_proj['path'])

            return redirect(url_for('start'))
        else:
            return redirect(url_for('home'))
 
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
    global process
    os.kill(process[pid].pid+1,signal.SIGTERM) #why should need I to sum 1 to pid???

    cstarted=started_model[pid]
    error=error_model.push_back(cstarted)
    error_model.save(error)
    started=started_model.remove(pid)
    started_model.save(started)    
    
    global used_gpus #only one job per gpu
    used_gpus.remove(int(cstarted['arguments']['use_cuda']))
    del process[pid]   
    return redirect(url_for('toogle_queue'))

#enqueue failed job-OK
@app.route("/experiment/enqueue/<int:pid>")
def enqueue(pid):
    experiments=experiments_model.push_back(error_model[pid])
    experiments_model.save(experiments)
    error=error_model.remove(pid)
    error_model.save(error)
    
    return redirect(url_for('start'))

@app.route("/about")
def about():
    return render_template('about.html', title='About')

