FROM inseefrlab/onyxia-vscode-python:py3.10.9

COPY requirements.txt .

RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update && \
    apt-get update && \
    apt-get install gdal-bin && \
    apt-get install libgdal-dev && \
    export CPLUS_INCLUDE_PATH=/usr/include/gdal && \
    export C_INCLUDE_PATH=/usr/include/gdal && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
