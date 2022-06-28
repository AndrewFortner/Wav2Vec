# Wav2Vec
Once you have downloaded all required files, naviate to the directory where you have saved the files and type: "pip install -r requirements.txt" on the command line to install required dependencies. This may take some time.

In order to run application for ASR using microphone:
1. Navigate to the directory where you installed Speec2Text.py, and type: "uvicorn Spech2Text:app --reload" Note that if this is your first time running the program some files will be installed for you. It may take some time as the model is quite large.
2. Wait a couple of seconds for the sever to startup. Once you see the message that says "INFO:     Application startup complete." then the application is ready to run.
3. Go to your local browser and type in "http://127.0.0.1:8000/begin" (You can alternatively make a GET request to this same URL). The application will now be listening for speech on your microphone.
4. Begin talking and a transcription of your speech will be printed to the console.
5. In order to close the application and see the full transcription, go to "http://127.0.0.1:8000/end" in either a browser or through a GET request, and the full transcription will be returned in JSON format.

In order to run application for transcription of a file:
1. Navigate to the directory where you installed Speec2Text.py, and type: "uvicorn Spech2Text:app --reload" Note that if this is your first time running the program some files will be installed for you. It may take some time as the model is quite large.
2. Wait a couple of seconds for the sever to startup. Once you see the message that says "INFO:     Application startup complete." then the application is ready to run.
3. Make a POST request to "http://127.0.0.1:8000" with integer "sr" as sample rate and wav file "file" as the file you want to trasncribe.
4. Wait for the file to finish processing. The full transcription will be returned in JSON format to http://127.0.0.1:8000. Typically transcription of a wav file takes about 1.5 seconds of processing for every 1 second of audio in the file.

  Why use Facebook's Wav2Vec over other open source ASR services? I have included my competitive analysis in the repository as "analysis.txt"
