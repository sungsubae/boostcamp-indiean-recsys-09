FROM python:3.8.5-slim-buster

RUN mkdir steamrec

COPY start.sh /steamrec/start.sh
COPY /backend /steamrec/backend
COPY /frontend /steamrec/frontend
COPY requirements.txt /steamrec/requirements.txt

RUN pip install --upgrade pip \ 
&& pip install -r /steamrec/requirements.txt

WORKDIR /steamrec

EXPOSE 8001
CMD ["sh", "start.sh"]

