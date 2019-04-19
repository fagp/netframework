from flask import Flask
import json
from flask_socketio import SocketIO
import argparse

parser = argparse.ArgumentParser(description='MeraMeraUI parameters')
parser.add_argument('--netfui', nargs='?', type=str, default='/home/', help='netfui root path')
parser.add_argument('--python', nargs='?', type=str, default='python', help='python path')
args = parser.parse_args()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'c4ce48d064ccc619888f3833257101d5'
app.config['paths']=json.load(open(app.config.root_path+'/static/config.json'))['paths']
app.config['root_path']=args.netfui
app.config['py_path']=args.python
socketio = SocketIO(app)

from netfui import routes
