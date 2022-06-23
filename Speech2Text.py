import threading
import io
import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, Form
import librosa

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft')

r = sr.Recognizer()
r.non_speaking_duration = 0.001
r.pause_threshold = .005
r.dynamic_energy_threshold = False

end = False
output = []
app = FastAPI()
@app.post("/")
async def upload(file: UploadFile, sr: int = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr)}

@app.get("/begin")
def begin():
    #Start script
    listening = threading.Thread(target = listen)
    listening.start()

@app.get("/end")
def stop():
    #End Script
    global end
    end = True
    print(toString(output))
    return {"text": toString(output)}

def process_wav(data, sample_rate):
    output.append('')
    audio, rate = librosa.load(data, sr = sample_rate)
    input_values = tokenizer(audio, sampling_rate = rate, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    return tokenizer.batch_decode(prediction)[0]

def process(data, thread_number):
        output.append('')
        # clip = AudioSegment.from_file(data)
        # x = torch.FloatTensor(clip.get_array_of_samples())
        audio, rate = librosa.load(data, sr = 16000)
        tokenized = tokenizer(audio, sampling_rate = rate, return_tensors = 'pt', padding = 'longest')
        inputs = tokenized.input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = tokenizer.batch_decode(tokens)
        output[thread_number] = text[0]
        print(text[0])

def listen():
    with sr.Microphone(sample_rate = 16000) as source:
        print("Say Something!")
        # r.adjust_for_ambient_noise(source)
        # print(r.energy_threshold)
        r.energy_threshold = 120
        thread_number = 0
        while not end:
            audio = r.listen(source)
            print("---------------")
            if end:
                return
            processing = threading.Thread(target = process, args = (io.BytesIO(audio.get_wav_data()), thread_number))
            processing.start()
            thread_number += 1

def toString(list):
    str = ''
    for i in list:
        if(i != ''):
            str += i + ' '
    return str.rstrip(str[-1])