import pygame
import io
import google.generativeai as genai 
import streamlit as st
import os
import speech_recognition as sr
from gtts import gTTS
import logging
import sys
import os
from llama_index.core import ServiceContext
# from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
# from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
import re
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from preprocess import preprocessing
from llama_index.core import Document
import qdrant_client
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

GOOGLE_API_KEY = "AIzaSyCHmmFgBimvvjn_ahULahxEnl76a_pXp8s"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key='AIzaSyCHmmFgBimvvjn_ahULahxEnl76a_pXp8s')
@st.cache_resource
def load_model():
    return genai.GenerativeModel('gemini-pro')
model = load_model()
chat = model.start_chat(history=[])
r = sr.Recognizer()
@st.cache_data
def initialize_mixer():
    pygame.mixer.init()
@st.cache_data
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

prompt_1 = "كمساعد افتراضي، يتمثل دورك كمساعد افتراضي في الإجابة عن الأسئلة التي أطرحها عليك حول محتوى التي أزودك بها"

def main():
    st.image("a.jpg")
    st.title("⚖️مستشاري : رفيقك في القانون، يستمع و يوجّه")

    # Add text with red color
    st.write('<span style="color:red">تنويه: يقدم مساعدنا القانوني معلومات وتوجيهات قانونية عامة، ولا يعتبر بديلاً عن النصيحة القانونية الشخصية. على الرغم من جهودنا للدقة، فإننا لا نضمن اكتمال أو تطبيقية المعلومات. استخدام هذا المساعد لا يؤسس لعلاقة محامي-موكل. يرجى استشارة محام للحصول على نصيحة متخصصة تتناسب مع حالتك الخاصة. نخلي مسؤوليتنا عن أي اعتماد على معلومات المساعد. من خلال استخدامه، فإنك توافق على هذه الشروط.</span>', unsafe_allow_html=True)
    initial_explanation_given = False
    if not initial_explanation_given:
        # If the initial explanation has not been given, provide it and set the flag to True
        response = chat.send_message(prompt_1,generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        temperature=0.0))
        #client = qdrant_client.QdrantClient(path="qdrant_gemini_3")
        #vector_store = QdrantVectorStore(client=client, collection_name="collection")
        model_1 = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
        model_2 = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
        @st.cache_resource 
        def load_embedding_model():
            return model_1
        
        embed_model = load_embedding_model()
        service_context = ServiceContext.from_defaults(
        llm=Gemini(api_key=GOOGLE_API_KEY), embed_model=embed_model
    )
        s_context = StorageContext.from_defaults(persist_dir="Embedding Store")
        
        #s_context = StorageContext.from_defaults(persist_dir=r"C:\Users\XPS\OneDrive - Ecole Centrale Casablanca\Bureau\Hackaton\Meta_Data_Extraction\model_2")
        
        vector_index = load_index_from_storage(s_context)
        index = vector_index
        engine = index.as_query_engine(service_context=service_context,similarity_top_k=3)
        initial_explanation_given = True
        st.text(" !جاهز ")
    if st.button("إبدأ المحادثة"):
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            st.text(" ... مستشاري يستمع ")
            stop_button = st.button("أغلق المحادثة")
            while True:  # Infinite loop for the conversation
                try:
                    audio2 = r.listen(source2)
                    MyText = r.recognize_google(audio2, language="ar-AR")  # Recognizing speech in English
                    MyText = MyText.lower()
                    st.text(f"السؤال: {MyText}")
                    RAG_response = engine.query(MyText+"in arabic")
                    st.text(f"السياق : {RAG_response}")
                    prompt = f'''You are an arabic law assistant, I want you to reformulate an the answer and explain clearly only in arabic this question :{MyText} based on this context {RAG_response}
develop your answer                     
'''
                    response = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=1,
                    temperature=0.0))

                    response =  response.text
                    response += "\n"
                    response += f"\n {engine.retrieve(MyText)[0].metadata} :طبقا للمادة" 
                    response += "من مدونة الاسرة "
                    response = response.replace('*','')
                    response = response.replace('-','')
                    # Displaying the response                  
                    st.text(f"جواب: {response}")
                    speak(response)
                    if stop_button:
                        break  # Exiting the loop if the user stops the conversation
                except sr.UnknownValueError:
                    st.text("Sorry, I couldn't understand the speech. Please try again.")
                except sr.RequestError:
                    st.text("There was an issue with the Google Speech Recognition service. Please check your internet connection or try again later.")

if __name__ == "__main__":
    main()
