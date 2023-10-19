
FROM inseefrlab/onyxia-vscode-pytorch:py3.11.4

COPY requirements.txt .

RUN sudo apt-get update
    sudo apt-get install python3-gdal -y
    sudo apt-get install libgdal-dev -y
    pip install -r requirements.txt

