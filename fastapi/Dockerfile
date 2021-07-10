FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html --default-timeout=100

COPY . /fastapi

EXPOSE 8000

RUN chmod +x ./start.sh

CMD ["./start.sh"]
