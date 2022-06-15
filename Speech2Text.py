import threading
import io
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
from pydub import AudioSegment
from fastapi import FastAPI

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-robust-ft-swbd-300h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-robust-ft-swbd-300h')

r = sr.Recognizer()
r.non_speaking_duration = 0.01
r.pause_threshold = .03

end = False
stop_sequence = 'STOP'
output = []
app = FastAPI()
@app.get("/begin")
def root():
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
    return {"raw": toString(output)}

def process(input, thread_number):
        data = io.BytesIO(input.get_wav_data())
        clip = AudioSegment.from_file(data)
        x = torch.FloatTensor(clip.get_array_of_samples())
        inputs = tokenizer(x, sampling_rate = 16000, return_tensors = 'pt', padding = 'longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = tokenizer.batch_decode(tokens)
        output[thread_number] = text[0]
        print(text[0])
        if text[0] == stop_sequence:
            print("stopping")
            global end
            end = True

def listen():
    with sr.Microphone(sample_rate = 16000) as source:
        print("Say something!")
        thread_number = 0
        while not end:
            audio = r.listen(source)
            if end:
                return
            output.append('')
            processing = threading.Thread(target = process, args = (audio, thread_number))
            processing.start()
            thread_number += 1

def toString(list):
    str = ''
    for i in list:
        if(i != ''):
            str += i + ' '
    return str