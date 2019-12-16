# -*- coding: utf-8 -*-

from os.path import join
DOC_ROOT_DIR = './build/html'

from flask import *
app = Flask(__name__, 
    static_folder=DOC_ROOT_DIR)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def return_index_linked_html(filename):
    return app.send_static_file(filename)

@app.route('/_static/<path:filename>')
def return_static(filename):
    return app.send_static_file(join('_static', filename))

