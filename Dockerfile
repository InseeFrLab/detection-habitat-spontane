FROM inseefrlab/onyxia-vscode-pytorch:py3.10.9

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
