import os, os.path, time
from flask import Flask, request
from werkzeug.utils import secure_filename
from ManTraNet import ManTraNet
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.nitf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = ManTraNet()

@app.route('/')
def index():
    return 'ManTraNet Service: Try uploading an image to /classify to use this model'

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    def is_file_allowed(filename):
        return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS
    file = request.files.get('file')
    if file == None or file == '':
        return 'No file given'
    elif not is_file_allowed(file.filename):
        return 'File not allowed. Received {0} but was expecting one of types {1}'.format(file.filename, ALLOWED_EXTENSIONS)
    else:
        filename = secure_filename(file.filename)
        save_destination = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_destination)
        _, mask, ptime = model.classify(save_destination)
        mask = (mask*256).astype(dtype='uint8')
        return {
            "mask": mask.tolist(),
            "time": ptime
        }
