# Voice Chat Demo

Voice chat demo using Whisper and Qwen2-Audio.

## Enviroments
```bash
conda create -n voice
conda activate voice
```

### Linux
```bash
sudo apt update
sudo apt install ffmpeg
pip install -r requirements_cpu.txt
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
streamlit run web_app.py
```

- 모델 change시 현재 memory cache 문제 해결 필요