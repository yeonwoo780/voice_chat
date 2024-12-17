import streamlit as st
from audiorecorder import audiorecorder
from tempfile import NamedTemporaryFile
import librosa
import base64
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForSpeechSeq2Seq, pipeline, AutoModel
import torch
from io import BytesIO
from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
from translate import _llm_translate
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import scipy

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
    elif model_name == "Qwen2-Audio-7B-Instruct":
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch_dtype)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    else:
        raise
    model = torch.compile(model)
    return model, processor

@st.cache_resource  # 👈 Add the caching decorator
def load_tts_model(lang):
    if lang == "arabic":
        tts_model_dir = "model/XTTS-v2/"
        tts_config = XttsConfig()
        tts_config.load_json(f"{tts_model_dir}config.json")
        tts_model = Xtts.init_from_config(tts_config)
        tts_model.load_checkpoint(tts_config, checkpoint_dir=f"{tts_model_dir}", eval=True)
        tts_model.to(device)
        tts_paths = "data/ttsvoice/"
        reference_audios = [tts_paths + i for i in os.listdir(f"{tts_paths}")]
        tts_processor = tts_model.get_conditioning_latents(audio_path=reference_audios)
    else:
        tts_processor = AutoProcessor.from_pretrained("suno/bark")
        tts_model = AutoModel.from_pretrained("suno/bark")
        tts_model.to(device)

    tts_model = torch.compile(tts_model)
    return tts_model, tts_processor

def language_dict(lang):
    translate_dict = {
        "korean": "ko",
        "japanese": "jp",
        "english": "en",
        "arabic": "ar"
    }
    return translate_dict[lang]

def bark_tts(target_text, tts_model, tts_processor):
    inputs = tts_processor(
        text = [target_text],
        return_tensors="pt",
    ).to(device)
    speech_values = tts_model.generate(**inputs, do_sample=True)
    sampling_rate = tts_model.generation_config.sample_rate
    speech_values = speech_values.cpu().numpy().squeeze()
    return speech_values, sampling_rate

def XTTS_tts(target_text, tts_model, tts_processor, lang):
    gpt_cond_latent, speaker_embedding = tts_processor
    speech_values = tts_model.inference(
        text=target_text,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        language=language_dict(lang),
        enable_text_splitting=True
    )
    sampling_rate = 24000
    speech_values = speech_values['wav']
    return speech_values, sampling_rate

def tts_inference(target_text, tts_model, tts_processor, lang):
    with NamedTemporaryFile(suffix=".mp3") as temp:
        file_name = temp.name
        if lang == "arabic":
            speech_values, sampling_rate = XTTS_tts(target_text, tts_model, tts_processor, lang)
        else:
            speech_values, sampling_rate = bark_tts(target_text, tts_model, tts_processor)
        scipy.io.wavfile.write(file_name, sampling_rate, speech_values)
        tts_embed = embed_audio(file_name)
        return tts_embed


def inference(audio, model_name, model, processor, lang):
    # Save audio to a file:
    with NamedTemporaryFile(suffix=".mp3") as temp:
        with open(f"{temp.name}", "wb") as f:
            f.write(audio.export().read())
        file_name = temp.name
        if model_name == "whisper-large-v3":
            asr = whisper_asr(file_name, model, processor, lang)
        elif model_name == "Qwen2-Audio-7B-Instruct":
            asr = qwen_asr(file_name, model, processor, lang)
        else:
            asr = ''
        embed = embed_audio(file_name)
    return asr, embed

def qwen_asr(file_name, model, processor, lang):
    audio, sr = librosa.load(f"{file_name}", sr=processor.feature_extractor.sampling_rate)
    conversation = [
        {'role': 'system', 'content': f'You are a helpful voice assistant.\nAnswer in the following language.\n\nLanguage: {lang}'},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": f"{file_name}"},
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return prediction

def whisper_asr(file_name, model, processor, lang):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(f"{file_name}", generate_kwargs={"language": lang})
    prediction =  result['text']
    return prediction

def embed_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    html = f'<audio controls>\n    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">\n    Your browser does not support the audio element.\n</audio>'
    return html

def clear_history():
    st.session_state.messages = []

def rewind():
    if st.session_state.messages:
        msg = st.session_state.messages.pop()
        while (msg.get('role', '') != 'user') and st.session_state.messages:
            msg = st.session_state.messages.pop()


LLM_MODELS = ["whisper-large-v3", "Qwen2-Audio-7B-Instruct"]
TTS_MODELS = ["korean", "japanese", "english", "arabic"]

# Streamlit
st.title("🎙️Voice ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    for _model in LLM_MODELS:
        load_model(_model)
    for _model in TTS_MODELS:
        load_tts_model(_model)
    clear_history()

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Audio Model", LLM_MODELS)
    lang = st.selectbox("Language", TTS_MODELS)
    model, processor = load_model(model_name)
    tts_model, tts_processor = load_tts_model(lang)
    st.header("Control")
    voice_embed = st.toggle('Show Audio', value=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("Rewind", on_click=rewind, use_container_width=True, type='primary')
    with btn_col2:
        st.button("Clear", on_click=clear_history, use_container_width=True)

# Display chat messages from history on app rerun
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        content = msg.get('content', '')
        if voice_embed:
            embed = msg.get('tts_embed', '')
            if i == (len(st.session_state.messages) - 1):
                embed = embed.replace('<audio controls>', '<audio controls autoplay>')
            content = '\n\n'.join([content, embed])
        st.markdown(content, unsafe_allow_html=True)

audio = audiorecorder("", "", key=f"audio_{len(st.session_state.messages)}")

# React to user input
if (prompt := st.chat_input("Your message")) or len(audio):
    # If it's coming from the audio recorder transcribe the message with whisper.cpp
    if model_name == "whisper-large-v3":
        if len(audio)>0:
            with st.spinner():
                prompt, embed = inference(audio, model_name, model, processor, lang)
            content = '\n\n'.join([prompt, embed]) if voice_embed else prompt
        else:
            content = prompt
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(content, unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": content
        })

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner():
                llm = ChatOpenAI(model="gpt-4o-2024-08-06")
                llm_response = llm.invoke(prompt).content
                st.markdown(llm_response)
                tts_embed = tts_inference(llm_response, tts_model, tts_processor, lang)
                st.markdown(
                    '\n\n'.join([tts_embed]),
                    unsafe_allow_html=True
                )
    else:
        if len(audio)>0:
            with st.spinner():
                llm_response, embed = inference(audio, model_name, model, processor, lang)
            content = embed
        else:
            content = prompt
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(content, unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": content
        })

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner():
                st.markdown(llm_response)
                tts_embed = tts_inference(llm_response, tts_model, tts_processor, lang)
                st.markdown(tts_embed, unsafe_allow_html=True)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": llm_response, "tts_embed": tts_embed})
    st.rerun()