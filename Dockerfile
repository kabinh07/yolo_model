FROM python:3.12

WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install -r requirements.txt


