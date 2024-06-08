import os

from openai import OpenAI
import base64
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo")
loader = CSVLoader(file_path="./laptops.csv")
data = loader.load()
vector = FAISS.from_documents(data, OpenAIEmbeddings())

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert sales assistant at a laptop store. Your job is to recommend a few laptops to the 
    customer based on their requirements.
    If the user is asking something other than buying laptops, let them know that this is a laptop store and purchasing 
    laptops is the only thing that can be done here. 
    Try to fit in the customer to a user persona like student, teacher, gamer, 
    business professional, researcher, teacher, casual everyday user, content creator etc. to recommend laptops. You 
    don't have to stick to the above personas strictly but this is a good guide. It is very important that you do not 
    mention to the user that you're trying to fit them into these personas. It is extremely important to ask user 
    questions to get more information about user's requirements and work pattern if you do not have enough 
    information to make an informed decision.It is absolutely mandatory to use only the below laptop products with 
    their detailed specification as mentioned in the inventory. Context:

<context>
{context}
</context>
         """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retriever.search_kwargs = {'k': 10}

history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the "
             "conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


def get_answer(question, chat_history):
    return retrieval_chain.invoke({"input": f"{question}", "chat_history": chat_history})["answer"]


def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript


def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)