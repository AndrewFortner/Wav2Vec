import threading
import io
import torch
from transformers import Wav2Vec2Processor, HubertForCTC, MBartForConditionalGeneration, MBart50TokenizerFast
import speech_recognition as sr
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import librosa
import soundfile

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft')

models, cfg, task = load_model_ensemble_and_task_from_hf_hub("facebook/fastspeech2-en-ljspeech", arg_overrides={"vocoder": "hifigan", "fp16": False})
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

r = sr.Recognizer()
r.non_speaking_duration = 0.001
r.pause_threshold = .005
r.dynamic_energy_threshold = False

#Supported languages and corresponding codes:
#  Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), 
#  Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN),
#  Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT),
#  Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO),
#  Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN),
#  Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL),
#  Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK),
#  Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF),
#  Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH),
#  Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)
TRANSLATE_LANG = 'en_XX'
end = False
output = []
num_starts = 0

app = FastAPI()
#English audio file into English text
@app.post("/transcribe-file")
async def transcribe(file: UploadFile, sr: int = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr, "en_XX")}

#English audio file into out_lang text
@app.post("/transcribe-file-translate")
async def translate_transcription(file: UploadFile, sr: int = Form(), out_lang: str = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr, out_lang)}

@app.post("/transcribe-file-en-es")
async def translate_transcription(file: UploadFile, sr: int = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr, "es_XX")}

#English speech into out_lang text
@app.post("/asr-live-translate")
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
    TRANSLATE_LANG = "en_XX"
    if num_starts % 2 == 1:
        end = False
        listening = threading.Thread(target = listen)
        listening.start()
    else:
        end = True
        return {"text": toString(output)}

@app.get("/asr-live-en-es")
def begin():
    #Start script
    global TRANSLATE_LANG
    global end
    global num_starts
    num_starts += 1
    TRANSLATE_LANG = "es_XX"
    if num_starts % 2 == 1:
        end = False
        listening = threading.Thread(target = listen)
        listening.start()
    else:
        end = True
        return {"text": toString(output)}

#TTS in_lang text -> out_lang audio file
@app.post("/tts-translate")
def synthesize_translation(text : str = Form(), in_lang: str = Form(), out_lang: str = Form()):
    #Translate
    if in_lang != out_lang:
        tokenizer.src_lang = in_lang
        text = text.replace('.', ',')
        encode = tokenizer(text, return_tensors = "pt")
        tokens = translation_model.generate(**encode, forced_bos_token_id=tokenizer.lang_code_to_id[out_lang])
        text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    #TTS
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
    audio = io.BytesIO()
    soundfile.write(audio, wav, rate, format = "WAV")
    audio.seek(0)
    return StreamingResponse(audio, media_type = "audio/wav")

#TTS English text -> English audio file
@app.post("/tts")
def synthesize(text : str = Form()):
    #TTS
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
    audio = io.BytesIO()
    soundfile.write(audio, wav, rate, format = "WAV")
    audio.seek(0)
    return StreamingResponse(audio, media_type = "audio/wav")

@app.post("/tts-en-to-es")
def tts_en_to_es(text : str = Form()):
    #Translate
    in_lang = "en_XX"
    out_lang = "es_XX"
    translation_tokenizer.src_lang = in_lang
    encode = translation_tokenizer(text, return_tensors = "pt")
    tokens = translation_model.generate(**encode, forced_bos_token_id=translation_tokenizer.lang_code_to_id[out_lang])
    text = translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(text)
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
def process_wav(data, sample_rate, out_lang):
    audio, rate = librosa.load(data, sr = sample_rate)
    input_values = tokenizer(audio, sampling_rate = rate, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    text = tokenizer.batch_decode(prediction)[0].lower()
    if(out_lang == 'en_XX'):
        return text
    translation_tokenizer.src_lang = 'en_XX'
    encode = translation_tokenizer(text, return_tensors = "pt")
    tokens = translation_model.generate(**encode, forced_bos_token_id=translation_tokenizer.lang_code_to_id[out_lang])
    translated = translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return translated

#ASR of microphone
def process(data, thread_number):
        global TRANSLATE_LANG
        output.append('')
        audio, rate = librosa.load(data, sr = 16000)
        tokenized = tokenizer(audio, sampling_rate = rate, return_tensors = 'pt', padding = 'longest')
        inputs = tokenized.input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = tokenizer.batch_decode(tokens)[0].lower()
        text+='.'
        if(TRANSLATE_LANG != 'en_XX' and text.count(' ') != 0):
            translation_tokenizer.src_lang = 'en_XX'
            encode = translation_tokenizer(text, return_tensors = "pt")
            tokens = translation_model.generate(**encode, forced_bos_token_id=translation_tokenizer.lang_code_to_id[TRANSLATE_LANG])
            text = translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
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
