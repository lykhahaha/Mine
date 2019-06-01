# jupyter-notebook --no-browser --port=8888
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O FILENAME

# screen on ubuntu
# apt-get update
# apt-get upgrade
# apt-get install screen
# type "screen". this will open up a new screen
# type: Ctrl + A, and then Ctrl + D. This will detach
# screen -ls | grep pts | cut -d. -f1 | awk '{print $1}' | xargs kill

# git clone https://github.com/chentinghao/download_google_drive.git
# python download_gdrive.py GoogleFileID /path/for/this/file/to/download/file.zip

# Kaggle for windows
python -m pip install kaggle
# copy .kaggle from User

# For Colab Kaggle
# !pip install kaggle --upgrade
# !mkdir /root/.kaggle
# %%writefile /root/.kaggle/kaggle.json
# {"username":"huongung155","key":"fd62aafc43796137f4e3bd2719ba7800"}
# !chmod 600 /root/.kaggle/kaggle.jsons

# kaggle competitions submit -c digit-recognizer -f submission.csv -m "First ever"
# kaggle competitions download -c digit-recognizer

# upload filet from GCP to Drive
# wget https://docs.google.com/uc?id=0B3X9GlR6EmbnWksyTEtCM0VfaFE&export=download
# mv uc\?id\=0B3X9GlR6EmbnWksyTEtCM0VfaFE gdrive
# chmod +x gdrive
# sudo install gdrive /usr/local/bin/gdrive
# gdrive list
# gdrive upload backups.tar.gz

# Connect Colab to Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# tensorboard on google colab
LOG_DIR = './logs'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# download for windows Powershell
# $client = new-object System.Net.WebClient
# $client.DownloadFile("http://www.xyz.net/file.txt","C:\tmp\file.txt")

# Set tensorflow for Keras
# set KERAS_BACKEND=tensorflow

# {
#     "floatx": "float32",
#     "epsilon": 1e-07,
#     "backend": "theano",
#     "image_data_format": "channels_last"
# }

# {
# 	"backend": "tensorflow",
# 	"floatx": "float32",
# 	"epsilon": 1e-07
# }

# Instal cuda
https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-together-with-9-2-on-Ubuntu-18-04-with-support-for-NVIDIA-20XX-Turing-GPUs-1236/
https://dmitry.ai/t/topic/33
tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz

# Install cuda on K2100M
Disable secure boot
sudo ubuntu-drivers autoinstall
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-390
sudo apt-get --purge remove xserver-xorg-video-nouveau
Then install cuda like above

sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*

# https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52

pip install tensorflow-gpu

# https://dwijaybane.wordpress.com/2017/07/23/setup-theano-with-cuda-support-on-ubuntu-16-04/
conda install pygpu
pip install Theano
echo -e "\n[nvcc]\nflags=-D_FORCE_INLINES\n" >> ~/.theanorc
# Open file “gedit ~/.theanorc” add edit the path to CUDA root:

[nvcc]
flags=-D_FORCE_INLINES

[cuda]
root = /usr/local/cuda-9.0
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
git checkout tags/v0.6.2 -b v0.6.2
mkdir Build
cd Build
# you can pass -DCMAKE_INSTALL_PREFIX=/path/to/somewhere to install to an alternate location
cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
sudo make install
cd ..
python setup.py build
sudo ldconfig

pip install Keras

nano ~/.keras/keras.json
# {
#     "floatx": "float32",
#     "epsilon": 1e-07,
#     "backend": "theano",
#     "image_data_format": "channels_last"
# }

# Install opencv 4 ubuntu
https://pydeeplearning.com/opencv/install-opencv3-with-anaconda-python3-6-on-ubuntu-18-04/
https://github.com/opencv/opencv/issues/12625#issuecomment-437042530
cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=OFF \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
      -D WITH_CUBLAS=ON \
      -D BUILD_OPENCV_PYTHON3=ON \
      -D PYTHON3_EXECUTABLE=~/anaconda3/bin/python \
      -D PYTHON3_INCLUDE_PATH=~/anaconda3/include/python3.7m \
      -D PYTHON3_LIBRARIES=~/anaconda3/lib/python3.7/site-packages \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_SHARED_LIBS=ON ..
make -j $(($(nproc) + 1))
sudo make install
sudo ldconfig


# Install xgboost
wget https://s3.amazonaws.com/pytorch/nccl_2.2.13-1%2Bcuda9.0_x86_64.txz
tar -xvf nccl_2.2.13-1+cuda9.0_x86_64.txz
sudo mkdir -p /usr/local/nccl-2.2
sudo cp -vRf nccl_2.2.13-1+cuda9.0_x86_64/* /usr/local/nccl-2.2
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=/usr/local/nccl-2.2
make -j4
cd ../python-package
sudo env "PATH=$PATH" python setup.py install

 conda install faiss-gpu cuda90 -c pytorch

 pip install cntk-gpu

 #MXNET
pip install mxnet-cu100
pip install numpy --upgrade

#LIGHTGBM
sudo apt-get install --no-install-recommends build-essential libboost-dev libboost-system-dev libboost-filesystem-dev
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM
mkdir build ; cd build
sudo apt install ocl-icd-opencl-dev
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)
cd ..; cd python-package/
sudo env "PATH=$PATH" python setup.py install --precompile

# Fastai
## for ubuntu
## if gpu compute compability > 5.
conda install -c pytorch pytorch-nightly cuda90
conda install -c fastai torchvision-nightly
conda install -c fastai fastai
## if gpu cc < 5, download http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl, then
pip install fastai

## for Windows
## if gpu compute compability > 5., just
# Downgrade pytorch from 0.4 by: conda install -c peterjc123 pytorch cuda90, or 0.4 (check in future b/c fastai does not support now)
# Finally, pip install fastai
## if gpu compute compability < 5., just download https://goo.gl/CxN8R1 or https://goo.gl/7fb8AA, then upgrade numpy (if nesscessary), then
# Download Shapely from https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
# Download Fiona from https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona

# Catboost
conda config --add channels conda-forge
conda install catboost

# Connect GCloud to Terminal
# Tao ssh: ssh-keygen -t rsa -f ~/.ssh/my-ssh-key -C quan (trong do quan: ten instance)
# Lay content trong my-ssh-key.pub, add content do vao ssh trong cloud instance
# Connect bang: ssh -i ~/.ssh/my-ssh-key quan@35.237.135.207 (quan: ten instance, 35.237.135.207: external IP)
# Ket thuc session: Ctrl D
