# get OS and install development tools
FROM ubuntu:latest
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install virtualenv
RUN apt -y install python3-pip

# define working directory
WORKDIR /metadb-setup

# install AgriSatPy
# Step 1: create a virtual environment
RUN virtualenv ${WORKDIR}/GeoPython
RUN bash -c 'source ${WORKDIR}/GeoPython/bin/activate'
# Step 2: COPY source code
COPY . .
# Step 3: run setup.py and install packages
RUN python3 setup.py bdist_wheel
RUN pip install rasterio==1.3a3
RUN pip install dist/*
RUN rm -rf dist/

# run AgriSatPy test modules
RUN pytest --log-file=${WORKDIR}/pytest.log

# create Database tables if they do not exist yet
# COPY . .
# RUN python3 ${WORKDIR}/scripts/setup_db.py

# give access to bash
CMD ["bash"]
