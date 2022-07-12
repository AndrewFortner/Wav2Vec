import threading
import io
from transformers import pipeline
import speech_recognition as sr
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import librosa
import soundfile

asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-xls-r-2b-22-to-16", feature_extractor="facebook/wav2vec2-xls-r-2b-22-to-16")

r = sr.Recognizer()
r.non_speaking_duration = 0.001
r.pause_threshold = .005
r.dynamic_energy_threshold = False


MAPPING = {
    "en": 250004,
    "de": 250003,
    "tr": 250023,
    "fa": 250029,
    "sv": 250042,
    "mn": 250037,
    "zh": 250025,
    "cy": 250007,
    "ca": 250005,
    "sl": 250052,
    "et": 250006,
    "id": 250032,
    "ar": 250001,
    "ta": 250044,
    "lv": 250017,
    "ja": 250012,
}

TRANSLATE_LANG = 'en'
end = False
output = []
num_starts = 0

app = FastAPI()
#English audio file into English text
@app.post("/transcribe-file")
async def transcribe(file: UploadFile, sr: int = Form()):
    global TRANSLATE_LANG
    data = io.BytesIO(await file.read())
    TRANSLATE_LANG = 'en'
    return {"text": process_wav(data, sr)}

#English audio file into out_lang text
@app.post("/transcribe-file-multilingual")
async def translate_transcription(file: UploadFile, sr: int = Form(), in_lang: str = Form()):
    data = io.BytesIO(await file.read())
    TRANSLATE_LANG = in_lang
    return {"text": process_wav(data, sr)}

#English speech into out_lang text
@app.post("/asr-live-multilingual")
async def translate(out_lang: str = Form()):
    global TRANSLATE_LANG
    global end
    global num_starts
    num_starts += 1
    TRANSLATE_LANG = out_lang
    if num_starts % 2 == 1:
        end = False
        listening = threading.Thread(target = listen)
        listening.start()
    else:
        end = True
        return {"text": toString(output)}

#English speech to english text
@app.get("/asr-live")
def begin():
    #Start script
    global TRANSLATE_LANG
    global end
    global num_starts
    num_starts += 1
    TRANSLATE_LANG = "en"
    if num_starts % 2 == 1:
        end = False
        listening = threading.Thread(target = listen)
        listening.start()
    else:
        end = True
        return {"text": toString(output)}

#TTS English text -> English audio file
@app.post("/tts")
def synthesize(text : str = Form()):
    #TTS
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub("facebook/fastspeech2-en-ljspeech", arg_overrides={"vocoder": "hifigan", "fp16": False})
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
    audio = io.BytesIO()
    soundfile.write(audio, wav, rate, format = "WAV")
    audio.seek(0)
    return StreamingResponse(audio, media_type = "audio/wav")

@app.post("/tts-es")
def tts_en_to_es(text : str = Form()):
    #TTS
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/tts_transformer-es-css10",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
    audio = io.BytesIO()
    soundfile.write(audio, wav, rate, format = "WAV")
    audio.seek(0)
    return StreamingResponse(audio, media_type = "audio/wav")

#ASR of file
def process_wav(data, sample_rate):
    audio, rate = librosa.load(data, sr = sample_rate)
    text = asr(audio, forced_bos_token_id=MAPPING[TRANSLATE_LANG]).lower()
    return text

#ASR of microphone
def process(data, thread_number):
        output.append('')
        audio, rate = librosa.load(data, sr = 16000)
        text = asr(audio, forced_bos_token_id=MAPPING[TRANSLATE_LANG]).lower()
        output[thread_number] = text
        print(text)

#Capture microphone audio
def listen():
    with sr.Microphone(sample_rate = 16000) as source:
        print("Say Something!")
        r.adjust_for_ambient_noise(source)
        thread_number = 0
        while not end:
            audio = r.listen(source)
            print("---------------")
            if end:
                return
            processing = threading.Thread(target = process, args = (io.BytesIO(audio.get_wav_data()), thread_number))
            processing.start()
            thread_number += 1

#Clean up output
def toString(list):
    str = ''
    for i in list:
        if(i != ''):
            str += i + ' '
    return str.rstrip(str[-1])
