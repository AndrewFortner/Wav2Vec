import pyttsx3
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[32].id) #Voice of Rishi

app = FastAPI()
@app.post("/")
def upload(text : str = Body()):
    engine.save_to_file(text, "tts.mp3")
    engine.runAndWait()
    return FileResponse("tts.mp3")