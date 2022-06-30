import threading
import io
import torch
from transformers import Wav2Vec2Processor, HubertForCTC, MBartForConditionalGeneration, MBart50TokenizerFast
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, Form
import librosa

tokenizer = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft')
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

app = FastAPI()
@app.post("/transcribe")
async def transcribe(file: UploadFile, sr: int = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr, "en_XX")}

@app.post("/translate-transcription")
async def translate_transcription(file: UploadFile, sr: int = Form(), out_lang: str = Form()):
    data = io.BytesIO(await file.read())
    return {"text": process_wav(data, sr, out_lang)}

@app.post("/translate-live")
async def translate(out_lang: str = Form()):
    global TRANSLATE_LANG
    global end
    end = False
    TRANSLATE_LANG = out_lang
    listening = threading.Thread(target = listen)
    listening.start()

@app.get("/transcribe-live")
def begin():
    #Start script
    global TRANSLATE_LANG
    global end
    TRANSLATE_LANG = "en_XX"
    end = False
    listening = threading.Thread(target = listen)
    listening.start()

@app.get("/end")
def stop():
    #End Script
    global end
    end = True
    print(toString(output))
    return {"text": toString(output)}

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
    translated = translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return translated

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
            text = translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)
        output[thread_number] = text
        print(text)

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

def toString(list):
    str = ''
    for i in list:
        if(i != ''):
            str += i + ' '
    return str.rstrip(str[-1])
