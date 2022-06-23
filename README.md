# Wav2Vec
Once you have downloaded all required files, naviate to the directory where you have saved the files and type: "pip install -r requirements.txt" on the command line to install required dependencies. This may take some time.
In order to run application after downloading all required files and installing dependencies:
1. Navigate to the directory where you installed Speec2Text.py, and type: "uvicorn Spech2Text:app --reload" Note that if this is your first time running the program some files will be installed for you. It may take some time as the model is quite large 
2. Wait a couple of seconds for the sever to startup. Once you see the message that says "INFO:     Application startup complete." then the application is ready to run.
3. Go to your local browser and type in "(http://127.0.0.1:8000/begin)" (You can alternatively make a GET request to this same URL). The application will now be listening for speech on your microphone.
4. Begin talking and a transcription of your speech will be printed to the console.
5. In order to close the application and see the full transcription, go to "http://127.0.0.1:8000/end" in either a browser or through a GET request, and the full transcription will be returned in JSON format.
6. You can now either close the terminal or use keyboard interrupt on the command line to close the server.

Why use Facebook's Wav2Vec over other open source ASR services?
Open-source competitive analysis for speech recognition:

       After searching through several open-source speech recognition services, I found that 
Facebook’s wav2vec 2.0 speech recognition models were the best choice for this project. In this analysis 
I will compare Facebook’s wav2vec 2.0, and Mozilla’s Deep Speech speech recognition models, because 
they are currently the two most advanced open-source speech recognition projects on the internet, to 
justify my decision. Both projects have extensive research in developing their speech recognition 
models, run locally without accessing any cloud service, and give quick and accurate results, and thus fit 
the criteria for this assignment.
        After refining the search down to wav2vec 2.0 and Deep Search, I ultimately decided that 
wav2vec 2.0 was the superior choice for this project. There are multiple factors that contribute to this. 
First of all, wav2vec 2.0 is significantly more accurate than Deep Speech, with a word error rate (WER) of 
just 1.5% when running all clean tests on labeled data of Librispeech (a large dataset used for training 
speech to text models). However, Deep Speech only achieves 6.56% WER when running the same tests. 
Meaning the Deep Speech model makes over 4 times more errors than the wav2vec 2.0’s model. 
Second, the Deep Speech model has been trained to recognize only 2 languages (English and Mandarin), 
while the wav2vec 2.0 model has been trained to recognize 53, which include but are not limited to 
English, Chinese, Cantonese, Spanish, Russian, French, German, Japanese, Bengali, and many more. 
Lastly, wav2vec 2.0 gets updated and researched far more than Deep Speech. The last release made on 
the Deep Speech github was over 18 months ago, and it is entirely based on a 2014 research paper. On 
the other hand, the team behind wav2vec2 is actively coming out with dramatically new and improved 
models based on brand new research, using the most cutting edge of machine learning and natural 
language processing technologies.
       Overall, Facebook’s wav2vec 2.0 project is leagues ahead of any other open-source project right 
now, and I believe that the momentum behind the project makes it the best candidate for a long term 
speech recognition service.
