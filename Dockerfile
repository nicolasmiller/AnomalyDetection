FROM ubuntu:utopic

RUN apt-get update

RUN apt-get -y install binutils gcc g++ build-essential wget \
               python python-dev python-pip python-support \
               python-cairo python-numpy python-scipy \
               git subversion curl r-base

RUN apt-get -y install libxml2-dev
RUN apt-get -y install libcurl4-openssl-dev
RUN apt-get -y install libcurl4-gnutls-dev
RUN apt-get -y install curl
RUN apt-get -y install libssl-dev


RUN pip install nose
RUN pip install mock
RUN pip install pandas

RUN apt-get -y autoremove

WORKDIR /root/AnomalyDetection
ENV PYTHONPATH /root/AnomalyDetection
