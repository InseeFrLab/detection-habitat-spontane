
FROM inseefrlab/onyxia-vscode-pytorch:py3.11.4

COPY requirements.txt .

RUN sudo apt-get update && \
    sudo apt-get install -y software-properties-common && \
    sudo apt-get update && \
    sudo add-apt-repository -y ppa:ubuntugis/ppa && \
    sudo apt-get update && \
    sudo apt-get install -y gdal-bin && \
    sudo apt-get install -y libgdal-dev && \
    export CPLUS_INCLUDE_PATH=/usr/include/gdal && \
    export C_INCLUDE_PATH=/usr/include/gdal && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
