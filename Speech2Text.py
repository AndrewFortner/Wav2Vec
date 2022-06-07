#from logging.config import listen
import threading
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-robust-ft-swbd-300h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-robust-ft-swbd-300h')

r = sr.Recognizer()
r.non_speaking_duration = 0.01
r.pause_threshold = .05

stop_sequence = 'STOP'
output = []

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
        #processing = threading.Thread(target = process, args = (audio,))
        global end
        end = False
        while not end:
            print("listening...")
            audio = r.listen(source)
            if end:
                del output[-1]
                for x in output:
                    if x == '':
                        output.remove(x)
                print(*output, sep = ' ')
                return
            print("finsihed listening")
            output.append('')
            processing = threading.Thread(target = process, args = (audio, thread_number))
            processing.start()
            #process(audio)
            thread_number += 1

listening = threading.Thread(target = listen)
listening.start()