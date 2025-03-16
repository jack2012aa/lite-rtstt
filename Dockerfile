# Dockerfile

FROM python:3.11-slim

WORKDIR /

COPY ./data_structure ./data_structure
COPY ./service ./service
COPY ./app.py ./app.py
COPY ./requirements.txt ./requirements.txt

# Install non-python dependencies
RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y portaudio19-dev

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8765

CMD ["python", "app.py"]