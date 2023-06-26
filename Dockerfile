FROM inseefrlab/onyxia-vscode-pytorch:py3.10.9

COPY requirements.txt .

RUN add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update && \
    sudo apt-get update && \
    sudo apt-get install gdal-bin && \
    sudo apt-get install libgdal-dev && \
    export CPLUS_INCLUDE_PATH=/usr/include/gdal && \
    export C_INCLUDE_PATH=/usr/include/gdal && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
