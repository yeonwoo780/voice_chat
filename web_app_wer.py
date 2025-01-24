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
from melo.api import TTS as meloTTS
import scipy
from Eval.utils import Evalinfo

# API KEY 정보로드
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@st.cache_resource  # 👈 Add the caching decorator
def load_model(model_name):
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.to(device)
    
    model = torch.compile(model)
    return model, processor

def inference(audio, model_name, model, processor, lang):
    # Save audio to a file:
    with NamedTemporaryFile(suffix=".mp3") as temp:
        with open(f"{temp.name}", "wb") as f:
            f.write(audio.export().read())
        file_name = temp.name
        model_name == "whisper-large-v3"
        asr = whisper_asr(file_name, model, processor, lang)
        embed = embed_audio(file_name)
    return asr, embed

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
        return f"""
<audio controls>
<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
</audio>
        """.strip()


TTS_MODELS = ["korean", "english"]

# Streamlit
st.title("WER Evaluation")

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("ASR model",["whisper-large-v3"])
    lang = st.selectbox("Language", TTS_MODELS)
    model, processor = load_model(model_name)
    if st.button("Clear Session State"):
        st.session_state.clear()
        st.rerun()

audio = audiorecorder("", "")
reference = st.text_area('WER 평가를 위해 말씀하신 내용을 직접 입력해주세요 (엔터키를 입력하지 마세요!)')

# Session state에 Evallist 초기화
if "Evallist" not in st.session_state:
    st.session_state.Evallist = []
    st.session_state.idx = 1
    st.session_state.Werlist = []

with st.form('Wer Eval', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if audio and reference and submitted:
        prediction, embed = inference(audio, model_name, model, processor, lang)
        evalinfo = Evalinfo(reference, prediction, lang)
        st.markdown(f"### 발화자 음성\n{embed}", unsafe_allow_html=True)
        st.markdown(f"- **원본 음성 텍스트**:    {evalinfo['reference']}", unsafe_allow_html=True)
        st.markdown(f"- **STT 모델 예측 텍스트**:    {evalinfo['prediction']}", unsafe_allow_html=True)
        st.markdown(f"- **원본 음성 텍스트 구절**:    {evalinfo['truth']}", unsafe_allow_html=True)
        st.markdown(f"- **예측 텍스트 구절**:    {evalinfo['hypothesis']}", unsafe_allow_html=True)
        st.markdown(f"- **S**:    {evalinfo['S']}", unsafe_allow_html=True)
        st.markdown(f"- **I**:    {evalinfo['I']}", unsafe_allow_html=True)
        st.markdown(f"- **D**:    {evalinfo['D']}", unsafe_allow_html=True)
        st.markdown(f"- **N**:    {evalinfo['N']}", unsafe_allow_html=True)
        st.markdown(f"- **WER**:    {evalinfo['WER']}", unsafe_allow_html=True)

        evalinfo['num'] = st.session_state.idx
        st.session_state.Werlist.append(evalinfo['WER'])
        st.session_state.Evallist.append(evalinfo)
        st.session_state.idx += 1
        torch.cuda.empty_cache()

if len(st.session_state.Evallist) > 0:
    st.dataframe(st.session_state.Evallist)
    TotalWER = round(sum(st.session_state.Werlist)/len(st.session_state.Werlist),2)
    st.markdown(f"### {lang} 통합 WER 오류율\n- {TotalWER}", unsafe_allow_html=True)