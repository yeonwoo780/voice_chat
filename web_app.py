import streamlit as st
from audiorecorder import audiorecorder
from tempfile import NamedTemporaryFile
import librosa
import base64
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from io import BytesIO
from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
# API KEY 정보로드
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@st.cache_resource  # 👈 Add the caching decorator
def load_model(precision):
        
    if precision == "whisper-large-v3":
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)
    elif precision == "Qwen2-Audio-7B":
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch_dtype)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")
    else:
        model = None
        processor = None
    return model, processor


def inference(audio, precision, model, processor):
    # Save audio to a file:
    with NamedTemporaryFile(suffix=".mp3") as temp:
        with open(f"{temp.name}", "wb") as f:
            f.write(audio.export().read())
        file_name = temp.name
        print(file_name)
        if precision == "Qwen2-Audio-7B":
            prediction = Qwen_ASR(file_name, model, processor)
        else:
            prediction = Whisper_ASR(file_name, model, processor)
    return prediction

def Qwen_ASR(file_name, model, processor):
    audio, sr= librosa.load(f"{file_name}", sr=processor.feature_extractor.sampling_rate)
    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|ko|>"
    inputs = processor(text=prompt, audios=audio, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return prediction

def Whisper_ASR(file_name, model, processor):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    # audio = librosa.load(BytesIO(f"{file_name}"))
    result = pipe(f"{file_name}", generate_kwargs={"language": "korean"})
    prediction =  result['text']
    return prediction


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

# Streamlit
with st.sidebar:
    audio = audiorecorder("Click to send voice message", "Recording... Click when you're done", key="recorder")
    st.title("🎙️Voice ChatBot")
    precision = st.selectbox("ASR model", ["whisper-large-v3","Qwen2-Audio-7B"])
    model, processor = load_model(precision)
    voice = st.toggle('Voice', value=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if (prompt := st.chat_input("Your message")) or len(audio):
    # If it's coming from the audio recorder transcribe the message with whisper.cpp
    if len(audio)>0:
        prompt = inference(audio, precision, model, processor)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"{prompt}"
    del prompt
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o-2024-08-06")
        llm_response = llm.invoke(response).content
        st.markdown(llm_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": llm_response})