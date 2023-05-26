FROM python:3.8.12-slim

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy model

COPY model model

# copy code

COPY hw2 hw2
COPY data data

ENV PYTHONPATH hw2

# standard cmd

CMD [ "python", "hw2/app.py" ]
