from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
from jiwer import compute_measures
import pandas as pd
import os
import re

wer_metric = load("wer")
def Eval(reference, prediction):
    wer = wer_metric.compute(references=[reference], predictions=[prediction])
    return wer

def replace_sentence(reference):
    """
    원본 Text 전처리. ', " 삭제 후 양 옆 공백 삭제
    """
    reference = reference.replace("'","").replace('"',"")
    reference = reference.strip()
    return reference

def strip_special_chars(reference):
    "특수문자 삭제"
    reference = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', reference)
    return reference

if __name__ == "__main__":
    Dataset.cleanup_cache_files
    dname = "mozilla-foundation/common_voice_17_0"
    dlang = "korean"
    print(f"Load Dataset: {dname}\nLanguage: {dlang}")
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "ko", split="validation", streaming=True)

    seed = 42
    print(f"Set Dataset seed: {seed}")

    tsetnum = 300
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
            language = "korean"
            file_path = audio_info['path']
            file_name = os.path.basename(file_path)
            reference = replace_sentence(test_dataset['sentence'])
            reference = strip_special_chars(reference)
            generate = pipe(audio_info, generate_kwargs={"language": "korean"})
            prediction = generate['text'].strip()
            prediction = strip_special_chars(prediction)
            measures = compute_measures(reference, prediction)
            substitutions = measures['substitutions']
            insertions = measures['insertions']
            deletions = measures['deletions']
            total_words = len(reference.split())
            wer = Eval(reference, prediction)
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
    fpath = "resultEval/korean_wer_eval.csv"
    df.to_csv(fpath, encoding="utf-8-sig", index=False)
    averages = df["WER"].mean()
    print(f"WER : {averages*100}%")
    print(f"Save Evaluate csv file: {fpath}")
    print("Evaluate Finish")