import pygame
import io
import google.generativeai as genai
import streamlit as st
import tempfile
import os
import pandas as pd
import time
import speech_recognition as sr
from gtts import gTTS
import logging
import sys
import pandas as pd
import os
from functions import get_wikipedia_text

from IPython.display import display
from IPython.display import Markdown

genai.configure(api_key='AIzaSyDE2osAYDuqBrX8p82HRUY9TXieQkl4qoo')
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

r = sr.Recognizer()
############################################### THE RAG CODE ##########################################################



from sentence_transformers import SentenceTransformer , util
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

paragraphs_embedd=model.encode(paragraphs)


#######################################################################################################################


def initialize_mixer():
    pygame.mixer.init()

def speak(text, lang='ar'):
    tts = gTTS(text=text, lang=lang)
    initialize_mixer()
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

st.title("Project Arabic Voice-Bot")
if st.button("Start Conversation"):
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2, duration=0.2)
        st.text("Listening for speech input...")
        # Asking if the user wants to continue the conversation
        stop_button = st.button("Stop Conversation")

        while True:  # Infinite loop for the conversation
            try:
                # Record audio
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2, language="ar-AR")  # Recognizing speech in Arabic
                MyText = MyText.lower()

                # Displaying vocal input on the interface
                st.text(f"Vocal Command: {MyText}")


                query_embedd=model.encode(MyText)
                search_result=util.semantic_search(query_embedd,paragraphs_embedd)
                most_similar_index = search_result[0][0]['corpus_id']
                correct_answer = paragraphs[most_similar_index]

                prompt = "you are an assistant chat bot, answer this quesion : " + MyText + " from  this paragraph : " + correct_answer


                # Generating response based on vocal command
                response = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=1,
                    temperature=0.0))
                response = response.text
                response = response.replace('*','')
                response = response.replace('-','')
                # Displaying the response
                st.text(f"paragraph : {prompt}")
                st.text(f"Response: {response}")

                # Speaking the response
                speak(response)

                if stop_button:
                    break  # Exiting the loop if the user stops the conversation

            except sr.UnknownValueError:
                st.text("Sorry, I couldn't understand the speech. Please try again.")
            except sr.RequestError:
                st.text("There was an issue with the Google Speech Recognition service. Please check your internet connection or try again later.")
