FROM ubuntu:22.04
RUN apt-get update \
 && apt-get install -y python3-pip build-essential git
RUN pip install -U pip
RUN pip install numpy scipy pytest ctypesgen wheel hatch
RUN pip install -U wheel
