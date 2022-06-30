import threading
import io
import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, Form
import librosa

# The model is trained on data with a sampling rate of 16000, so it would be ideal to provide
# audio with a sampling rate of 16000
tokenizer = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft')

r = sr.Recognizer()
r.non_speaking_duration = 0.001
r.pause_threshold = .005
r.dynamic_energy_threshold = False

end = False
output = []
num_starts = 0

app = FastAPI()
@app.post("/transcribe-file")
async def transcribe(file: UploadFile, sr: int = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr)}

@app.get("/transcribe-live")
def begin():
    #Start script
    global num_starts
    global end
    num_starts += 1
    if num_starts % 2 == 1:
        end = False
        listening = threading.Thread(target = listen)
        listening.start()
    else:
        end = True
        return {"text": toString(output)}

def process_wav(data, sample_rate):
    audio, rate = librosa.load(data, sr = sample_rate)
    input_values = tokenizer(audio, sampling_rate = rate, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    text = tokenizer.batch_decode(prediction)[0].lower()
    return text

def process(data, thread_number, sample_rate):
        output.append('')
        audio, rate = librosa.load(data, sr = sample_rate)
        tokenized = tokenizer(audio, sampling_rate = rate, return_tensors = 'pt', padding = 'longest')
        inputs = tokenized.input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = tokenizer.batch_decode(tokens)[0].lower()
        output[thread_number] = text
        print(text)

def listen():
    #Let the microphone have a sampling rate of 16000
    with sr.Microphone(sample_rate = 16000) as source:
        print("Say Something!")
        #r.adjust_for_ambient_noise(source)
        r.enery_threashold = 120
        thread_number = 0
        while not end:
            audio = r.listen(source)
            print("---------")
            if end:
                return
            processing = threading.Thread(target = process, args = (io.BytesIO(audio.get_wav_data()), thread_number, 16000))
            processing.start()
            thread_number += 1

def toString(list):
    str = ''
    for i in list:
        if(i != ''):
            str += i + ' '
    return str.rstrip(str[-1])
