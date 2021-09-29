FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app

COPY ./src .

RUN git clone https://github.com/huggingface/transformers
WORKDIR /app/transformers
RUN pip install .

RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

