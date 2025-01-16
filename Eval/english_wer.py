from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
from jiwer import compute_measures
import pandas as pd
import os
import re
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

wer_metric = load("wer")
def Eval(reference, prediction):
    wer = wer_metric.compute(references=[reference], predictions=[prediction])
    return wer

def english_normalize(text):
    norm_text = normalizer(text)
    return norm_text

if __name__ == "__main__":
    Dataset.cleanup_cache_files
    dname = "mozilla-foundation/common_voice_17_0"
    dlang = "english"
    print(f"Load Dataset: {dname}\nLanguage: {dlang}")
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="validation", streaming=True)

    seed = 42
    print(f"Set Dataset seed: {seed}")

    tsetnum = 1000
    print(f"Get Test Dataset: {tsetnum}")
    test_datasets = dataset.shuffle(seed=seed).take(tsetnum)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    mname = "openai/whisper-large-v3"
    print(f"Load Model: {mname}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        mname, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(mname)
    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    Eval_list = []
    
    n = 0 
    print("Load 30 test datasets with 0 downvotes")
    for test_dataset in test_datasets:
        if test_dataset['down_votes'] == 0 and test_dataset['up_votes'] > 3:
            audio_info = test_dataset['audio']
            language = "english"
            file_path = audio_info['path']
            file_name = os.path.basename(file_path)
            reference = english_normalize(test_dataset['sentence'])
            generate = pipe(audio_info, generate_kwargs={"language": "english"})
            prediction = generate['text'].strip()
            prediction = english_normalize(prediction)
            measures = compute_measures(reference, prediction)
            substitutions = measures['substitutions']
            insertions = measures['insertions']
            deletions = measures['deletions']
            total_words = len(reference.split())
            wer = Eval(reference, prediction)
            wer = round(wer,5)
            Eval_list.append({
                'num': n+1,
                'language': language,
                'file_path': file_path,
                'file_name': file_name,
                'reference': reference,
                'prediction': prediction,
                'S' : substitutions,
                'I' : insertions,
                'D' : deletions,
                'N' : total_words,
                'WER' : wer,
            })
            print(f"num: {n+1}  /  wer:{wer}")
            torch.cuda.empty_cache()
            n += 1
        if n == 30:
            break

    df = pd.DataFrame(Eval_list)
    fpath = "resultEval/english_wer_eval.csv"
    df.to_csv(fpath, encoding="utf-8-sig", index=False)
    averages = round(df["WER"].mean(), 5)
    print(f"WER : {averages*100}%")
    print(f"Save Evaluate csv file: {fpath}")
    print("Evaluate Finish")