FROM inseefrlab/onyxia-jupyter-python:py3.10.8

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
