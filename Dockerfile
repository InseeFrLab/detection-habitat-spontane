FROM inseefrlab/onyxia-vscode-python:py3.10.4

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
