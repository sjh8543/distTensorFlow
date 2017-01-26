#all images which I will develop is based on ubuntu:16.04
FROM ubuntu:16.04
#install dependency modules
RUN apt-get update && apt-get -y install git python-pip python2.7

#tensorflow install with pip. current image use tensorflow version 0.12
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

#create runtiem directory that is root directroy for running application in this image 
RUN mkdir /home/runtime

COPY SimpleDistServer.py /home/runtime
COPY DistributedSimpleNeuralNetwork.py /home/runtime 

