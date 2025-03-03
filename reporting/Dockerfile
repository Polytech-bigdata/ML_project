FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]