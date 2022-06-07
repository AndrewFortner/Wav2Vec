import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment



tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')


r = sr.Recognizer()

with sr.Microphone(sample_rate = 16000) as source:
    print("Say something!")
    while True:
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data)
        x = torch.FloatTensor(clip.get_array_of_samples())
        inputs = tokenizer(x, sampling_rate = 16000, return_tensors = 'pt', padding = 'do_not_pad').input_values
        logits = model(inputs).logits
        print(logits)
        tokens = torch.argmax(logits, axis = -1)
        print(tokens)
        text = tokenizer.batch_decode(tokens)

        print('You said: ', str(text).lower())