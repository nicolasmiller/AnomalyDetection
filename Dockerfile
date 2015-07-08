FROM ubuntu:utopic

RUN apt-get update

RUN apt-get -y install binutils gcc g++ build-essential wget \
               python python-dev python-pip python-support \
               gfortran \
               libopenblas-base libopenblas-dev \
               python-cairo python-numpy python-scipy \
               git subversion curl r-base

RUN apt-get -y install libxml2-dev
RUN apt-get -y install libcurl4-openssl-dev
RUN apt-get -y install libcurl4-gnutls-dev
RUN apt-get -y install curl
RUN apt-get -y install libssl-dev

RUN pip install numpy

#RUN pip install git+https://github.com/nicolasmiller/pyloess.git

RUN pip install nose
RUN pip install mock
RUN pip install pandas
RUN pip install pytz
RUN pip install statsmodels
RUN pip install rpy2

RUN apt-get -y autoremove

WORKDIR /root/AnomalyDetection
