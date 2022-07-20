FROM python:3.8.10

COPY ./asr_tts.py /
COPY ./requirements.txt /

WORKDIR /

RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN pip install pyaudio
RUN apt-get install libsndfile1-dev -y
RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "asr_tts:app", "--host=0.0.0.0"]
