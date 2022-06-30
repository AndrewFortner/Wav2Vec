from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import soundfile
import io

#Text to speech model
models, cfg, task = load_model_ensemble_and_task_from_hf_hub("facebook/fastspeech2-en-ljspeech", arg_overrides={"vocoder": "hifigan", "fp16": False})
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

#REST endpoint 
#Post form with name = "text" value = (text to synehtsize)
app = FastAPI()
@app.post("/")
def upload(text : str = Form()):
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
    audio = io.BytesIO()
    soundfile.write(audio, wav, rate, format = "WAV")
    audio.seek(0)
    return StreamingResponse(audio, media_type = "audio/wav")
