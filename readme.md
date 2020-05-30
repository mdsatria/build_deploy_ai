# Build dan Deploy Deep Feed Forward Network
[![](https://img.shields.io/badge/python-3.6%2B-green.svg)]()

Repositori ini menjelaskan bagaimana cara membangun dan mendeploy model Deep Learning menggunakan Tensorflow, Numpy dan Flask dengan bahasa Python untuk mengenali tulisan tangan berupa angka.

## Build
Untuk membuild model, Anda dapat membuka folder <a href="https://github.com/mdsatria/build_deploy_ai/tree/master/build">**build**</a>. Pada folder build disediakan script Python untuk membuat model dan notebook yang menjelaskan bagaimana membangun model tersebut. Anda bisa menjalankan notebook langsung dari browser Anda untuk membangun model dengan Google Colab. Buka link berikut untuk mencoba membangun model AI Anda sendiri.
<a href="https://colab.research.google.com/github/mdsatria/build_deploy_ai/blob/master/build/Building%20Deep%20Learning%20Model%20dengan%20Google%20Colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Deploy
Pada folder <a href="https://github.com/mdsatria/build_deploy_ai/tree/master/deploy">**deploy**</a> tersedia app sederhana dengan framework Flask untuk mendeploy model Deep Learning pada website.
Untuk menjalankan aplikasi secara lokal ikuti perintah berikut :
```shell
# 1. Pertama, clone repo ini
$ git clone https://github.com/mdsatria/build_deploy_ai.git
$ cd build_deploy_ai/deploy

# 2. Install Python library
$ pip install -r requirements.txt

# 3. Run app
$ python app.py
```
Buka http://localhost:5000