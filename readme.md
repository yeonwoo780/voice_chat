# Voice Chat Demo

Voice chat demo using Whisper and Qwen2-Audio.

## Enviroments
```bash
conda create -n voice
conda activate voice
```

### Linux
- cpu
```bash
sudo apt update
sudo apt install ffmpeg
sudo apt install portaudio19-dev
sudo apt install gcc
pip install audiorecorder
pip install -r requirements_cpu.txt
```

- gpu
```bash
sudo apt update
sudo apt install ffmpeg
sudo apt install portaudio19-dev
sudo apt install gcc
pip install audiorecorder
pip install -r requirements.txt
```


### MacOS
```zsh
brew install ffmpeg
brew install portaudio
pip install audiorecorder ffmpeg-python
pip install -r requirements_macos.txt
```

## Run

```bash
streamlit run web_app_asr.py
streamlit run web_app_chat.py
```

## tts model 검토
### bark [url](https://huggingface.co/suno/bark)

- sample notebooks[notebooks](notebooks/tts/bark.ipynb)

**language 지원**
- 한국어 O
- 일본어 O
- 아랍어 X

**tts 성능**
- 어색하지 않은 것 같다.


- 모델 change시 현재 memory cache 문제 해결 필요