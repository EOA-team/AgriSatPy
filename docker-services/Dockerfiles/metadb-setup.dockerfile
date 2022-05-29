FROM python:3.9-slim-buster

WORKDIR /metadb-setup

# install AgriSatPy
COPY install.sh install.sh
RUN ./install.sh

COPY . .

# create Database tables if they not exist yet
COPY ./scripts/setup_db.py /
RUN python3 ${WORKDIR}/setup_db.py
