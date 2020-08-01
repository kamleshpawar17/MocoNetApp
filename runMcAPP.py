import os
import shutil
import glob
import zipfile
from customPythonFunctions import readDicomFrmFolder
import nibabel as nb
import numpy as np

from flask import Flask, request, render_template, url_for, redirect, send_file
from gevent.pywsgi import WSGIServer
from DLMotionCorrection import DL_MC


mc_obj = DL_MC(isin3ch=True, isout3ch=False, mu_dst=0.5, sigma_dst=1 / 6., noise_frac=0.03)
dirname = './uploads/'
out_dir = './uploads/dicom_mc/'


def run_mc():
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mc_obj(dirname, out_dir)
    return 0


def zip_dir():
    zipf = zipfile.ZipFile('./uploads/dicom_mc.zip', 'w', zipfile.ZIP_DEFLATED)
    fnames = glob.glob('./uploads/dicom_mc/*.dcm')
    for f in fnames:
        fname = '/dicom_mc/' + f.split('/')[-1]
        zipf.write(f, fname)
    zipf.close()


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def fileFrontPage():
    return render_template('index.html')


@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    if request.method == 'POST':
        if os.path.exists('./uploads/'):
            shutil.rmtree('./uploads/')
            os.mkdir('./uploads')
        else:
            os.mkdir('./uploads')
        files = request.files.getlist("file")
        basepath = os.path.dirname(__file__)
        for f in files:
            if f.filename != '':
                f.save(os.path.join(basepath, 'uploads', f.filename))
        # ---- write nifiti for display ---- 3
        img, ds = readDicomFrmFolder('./uploads')
        imgInpObj = nb.Nifti1Image(img, np.eye(4))
        nb.save(imgInpObj, './static/data/recievedFilesNifti/input.nii')
        return redirect(url_for('fileFrontPage'))


@app.route("/process", methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        try:
            run_mc()
            zip_dir()
        except:
            return str('Failed to Process, check that all dicom files were uploaded')
    return render_template('index.html')


@app.route("/return_mc_zip", methods=['GET', 'POST'])
def return_mc_zip():
    try:
        return send_file('./uploads/dicom_mc.zip', attachment_filename='dicom_mc.zip')
    except Exception:
        return str('file not found')


if __name__ == '__main__':
    ip = '127.0.0.1'
    print('Running on IP - http://127.0.0.1:5000')
    http_server = WSGIServer((ip, 5000), app)
    http_server.serve_forever()
