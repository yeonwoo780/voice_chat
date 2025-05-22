import os
os.chdir("../")
print(os.getcwd())

import warnings
warnings.filterwarnings('ignore')

from pydub import AudioSegment

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

folder = "./chat/"
folder_list = os.listdir(folder)
objs = [folder + i for i in folder_list]

response = []
for obj in objs:
    save_folder = "./voice_chat/data/noise/"
    basename = os.path.basename(obj)
    text = os.path.splitext(basename)[0]
    tar_obj = save_folder + text + ".mp3"

    audio = AudioSegment.from_file(obj, format="m4a")
    audio.export(tar_obj, format="mp3")

    target_audio_data = pipe(tar_obj)

    chunks = target_audio_data['chunks']
    for k in chunks:
        timestamp = k['timestamp']
        starttime = timestamp[0]
        endtime = timestamp[1]
        content = k['text']
        response.append({
            'File': basename,
            'Start': starttime,
            'End': endtime,
            'pred': content
        })
    torch.cuda.empty_cache()

df = pd.DataFrame(response)
fpath = "./voice_chat/resultEval/noise_test.csv"
df.to_csv(fpath, encoding="utf-8-sig", index=False)

print(f"Save Evaluate csv file: {fpath}")