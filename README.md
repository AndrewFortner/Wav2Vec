# Running The Program
Once you have downloaded all .py files, naviate to the directory where you have saved the files and type: "pip install -r requirements.txt" on the command line to install required dependencies. This may take some time.

In order to run the application for ASR using microphone:
1. Navigate to the directory where you installed asr_tts.py, and type: "uvicorn asr_tts:app"
2. Wait a couple of seconds for the sever to startup. Once you see the message that says "INFO:     Application startup complete." then the application is ready to run.
3. Go to your local browser and type in "http://127.0.0.1:8000/asr-live" (You can alternatively make a GET request to this same URL). The application will now be listening for speech on your microphone.
4. Begin talking and a transcription of your speech will be printed to the console.
5. In order to close the application and see the full transcription, reload "http://127.0.0.1:8000/asr-live" in either a browser or through a GET request, and the full transcription will be returned in JSON format.

In order to run application for transcription of a file:
1. Navigate to the directory where you installed asr_tts.py, and type: "uvicorn asr_tts:app"
2. Wait a couple of seconds for the sever to startup. Once you see the message that says "INFO:     Application startup complete." then the application is ready to run.
3. Make a POST request to "http://127.0.0.1:8000/transcribe-file" using a form with integer "sr" as sample rate and wav file "file" as the file you want to trasncribe.
4. Wait for the file to finish processing. The full transcription will be returned in JSON format to http://127.0.0.1:8000/transcribe-file. Typically transcription of a wav file takes about 1.5 seconds of processing for every 1 second of audio in the file.

In order to run application for text to speech:
1. Navigate to the directory where you installed asr_tts.py, and type: "uvicorn asr_tts:app"
2. Wait a couple of seconds for the sever to startup. Once you see the message that says "INFO:     Application startup complete." then the application is ready to run.
3. Make a POST request to "http://127.0.0.1:8000/tts" using a form with string "text" as the text you want to synthesize.
4. Wait for the file to finish processing. The full transcription will be returned in JSON format to http://127.0.0.1:8000/synthesize.
