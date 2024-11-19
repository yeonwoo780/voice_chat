import streamlit as st
from audiorecorder import audiorecorder
from tempfile import NamedTemporaryFile
import librosa
import base64
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForSpeechSeq2Seq, pipeline
import torch
from io import BytesIO
from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
# API KEY 정보로드
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@st.cache_resource  # 👈 Add the caching decorator
def load_model(model_name):
    if model_name == "whisper-large-v3":
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)
        # model = None
        # processor = None
    elif model_name == "Qwen2-Audio-7B-Instruct":
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch_dtype)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    else:
        model = None
        processor = None
    return model, processor


def inference(audio, model_name, model, processor):
    # Save audio to a file:
    with NamedTemporaryFile(suffix=".mp3") as temp:
        with open(f"{temp.name}", "wb") as f:
            f.write(audio.export().read())
        file_name = temp.name
        print(file_name)
        if model_name == "whisper-large-v3":
            asr = whisper_asr(file_name, model, processor)
        elif model_name == "Qwen2-Audio-7B-Instruct":
            asr = qwen_asr(file_name, model, processor)
        else:
            asr = ''
        embed = embed_audio(file_name)
    return asr, embed

def qwen_asr(file_name, model, processor):
    audio, sr = librosa.load(f"{file_name}", sr=processor.feature_extractor.sampling_rate)
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": f"{file_name}"}
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return prediction

def whisper_asr(file_name, model, processor):
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

def embed_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"""
<audio controls>
<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
</audio>
        """.strip()

def clear_history():
    st.session_state.messages = []

def rewind():
    if st.session_state.messages:
        msg = st.session_state.messages.pop()
        while (msg.get('role', '') != 'user') and st.session_state.messages:
            msg = st.session_state.messages.pop()


# Streamlit
st.title("🎙️Voice ChatBot")

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Audio Model", ["whisper-large-v3", "Qwen2-Audio-7B-Instruct"])
    model, processor = load_model(model_name)
    st.header("Control")
    voice_embed = st.toggle('Show Audio', value=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("Rewind", on_click=rewind, use_container_width=True, type='primary')
    with btn_col2:
        st.button("Clear", on_click=clear_history, use_container_width=True)

# Initialize chat history
if "messages" not in st.session_state:
    clear_history()

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg.get('content', '')
        if voice_embed:
            embed = msg.get('embed', '')
            content = '\n\n'.join([content, embed])
        st.markdown(content, unsafe_allow_html=True)

audio = audiorecorder("", "", key=f"audio_{len(st.session_state.messages)}")

# React to user input
if (prompt := st.chat_input("Your message")) or len(audio):
    # If it's coming from the audio recorder transcribe the message with whisper.cpp
    if model_name == "whisper-large-v3":
        if len(audio)>0:
            with st.spinner():
                prompt, embed = inference(audio, model_name, model, processor)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(
                '\n\n'.join([prompt, embed]) if voice_embed else prompt,
                unsafe_allow_html=True
            )
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "embed": embed
        })

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-4o-2024-08-06")
            llm_response = llm.invoke(prompt).content
            st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
    else:
        if len(audio)>0:
            with st.spinner():
                prompt, embed = inference(audio, model_name, model, processor)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(
                '\n\n'.join([embed]) if voice_embed else prompt,
                unsafe_allow_html=True
            )
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "embed": embed
        })

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": prompt})