FROM python:3

WORKDIR C:\Users\lenovo\repo\Machine_Learning_Project\Machine_Learning_Project

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py

CMD flask run --host=0.0.0.0