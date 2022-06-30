from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import soundfile
import io

#Text to speech model
models, cfg, task = load_model_ensemble_and_task_from_hf_hub("facebook/fastspeech2-en-ljspeech", arg_overrides={"vocoder": "hifigan", "fp16": False})
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

#Text to text model
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

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

#Given text, translate from the input language to the output language and speak it.
app = FastAPI()
@app.post("/")
def upload(text : str = Form(), in_lang: str = Form(), out_lang: str = Form()):
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
