import threading
import io
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File
import librosa

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-robust-ft-swbd-300h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-robust-ft-swbd-300h')

r = sr.Recognizer()
r.non_speaking_duration = 0.01
r.pause_threshold = .03

end = False
output = []
app = FastAPI()
@app.post("/")
async def upload(file: UploadFile):
    data = io.BytesIO(await file.read())
    process_wav(data, 0)
    return {"text": toString(output)}

@app.post("/begin")
def begin():
    #Start script
    listening = threading.Thread(target = listen)
    listening.start()

@app.get("/end")
def stop():
    #End Script
    global end
    end = True
    print("Stopping")
    print(toString(output))
    return {"text": toString(output)}

def process_wav(data, thread_number):
    output.append('')
    audio, rate = librosa.load(data, sr = 16000)
    input_values = tokenizer(audio, sampling_rate = rate, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    output[thread_number] = tokenizer.batch_decode(prediction)[0]
    print("Done processing wav")

def process(data, thread_number):
        output.append('')
        clip = AudioSegment.from_file(data)
        x = torch.FloatTensor(clip.get_array_of_samples())
        inputs = tokenizer(x, sampling_rate = 16000, return_tensors = 'pt', padding = 'do_not_pad').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = tokenizer.batch_decode(tokens)
        output[thread_number] = text[0]
        print(text[0])
        print("done processing")

def listen():
    with sr.Microphone(sample_rate = 16000) as source:
        print("Say something!")
        thread_number = 0
        while not end:
            audio = r.listen(source)
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