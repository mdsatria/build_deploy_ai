# Flask untuk web interface
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
# TensorFlow dan tf.keras untuk AI
from tensorflow.keras.models import load_model
# Numpy untuk pemrosesan matriks
import numpy as np
# PIL untuk pengolahan image input
from PIL import Image
# utility
import re
import base64
import os
import sys

# inisialisasi Flask app
app = Flask(__name__)

# Tentukan path weight dan model yang sudah ditrain
MODEL_PATH = 'model/model.h5'
# Load model
model = load_model(MODEL_PATH)
#model._make_predict_function()         
# Msg ke server, model berhasil di-load
print('Model loaded. Start serving...')

# decoding image dari base64 ke representasi png
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
	#initModel()
	# render out pre-built HTML file di halaman index
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	# saat method dipanggil, angka yang digambar user akan dikonversi ke image. lakukan inference dengan model. kembalikan hasil klasifikasi
	# ambil raw data
	imgData = request.get_data()
	print('imgData :- ',type(imgData))
	# encode ke image format
	convertImage(imgData)
	print ("image convert stage")
	# resize png dan convert ke grayscale
	n_size = 28
	x = np.array(Image.open('output.png').resize((n_size, n_size)).convert('L'))
	print(x.dtype)
	# image invert
	x = np.invert(x)
	# normalisasi nilai piksel
	x = x.astype(np.float32)
	x /= 255
	# reshape image ke dimensi yang dibutuhkan model
	x = x.reshape(1,-1)
	print ("data prep stage")
	# prediksi
	out = model.predict(x)
	response = np.array_str(np.argmax(out,axis=1))	
	return response
	
if __name__ == "__main__":
	# tentukan port mana app akan berjalan
	port = int(os.environ.get('PORT', 5000))
	# locally run app
	app.run(host='0.0.0.0', port=port)
	# optional if we want to run in debugging mode
	app.run()