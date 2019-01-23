import json
from threading import Lock

mutex = Lock()

class model():
    def __init__(self,json_path):
        self.model_path=json_path
        self.last_index=-1

    def list_all(self):
        mutex.acquire()
        try:
            done=json.load(open(self.model_path))
        except:
            done=dict()
        mutex.release()
        return done

    def remove(self,id):
        done=self.list_all()
        del done[str(id)]
        return done

    def push_back(self,obj):
        done=self.list_all()
        if len(list(done.keys()))>0:
            newid=max([int(k) for k in list(done.keys())])+1
        else:
            newid=0
        done[str(newid)]=obj  
        self.last_index=str(newid)
        return done  
    
    def __getitem__(self,item):
        done=self.list_all()
        if str(item)=="-1":
            item= list(done.keys())[0]
        return done[str(item)]

    def save(self,objs):
        mutex.acquire()
        json.dump(objs,open(self.model_path,'w'))
        mutex.release()

    def enable(self,id):
        done=self.list_all()
        done[str(id)]['available']='True'
        self.save(done)

    def disable(self,id):
        done=self.list_all()
        done[str(id)]['available']='False'
        self.save(done)