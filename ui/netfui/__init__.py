from flask import Flask
import json
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'c4ce48d064ccc619888f3833257101d5'
app.config['paths']=json.load(open(app.config.root_path+'/static/config.json'))['paths']
socketio = SocketIO(app)

from netfui import routes
