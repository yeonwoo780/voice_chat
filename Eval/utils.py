from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
from jiwer import compute_measures
import pandas as pd
import os
import re
from Eval import korean_wer, english_wer

wer_metric = load("wer")
def get_eval(reference, prediction):
    wer = wer_metric.compute(references=[reference], predictions=[prediction])
    wer = round(wer,5)
    wer = wer * 100
    return wer

def replace_korean(reference, prediction):
    reference = korean_wer.replace_sentence(reference)
    reference = korean_wer.strip_special_chars(reference)
    prediction = prediction.strip()
    prediction = korean_wer.strip_special_chars(prediction)
    return reference, prediction

def replace_english(reference, prediction):
    reference = english_wer.english_normalize(reference)
    prediction = prediction.strip()
    prediction = english_wer.english_normalize(prediction)
    return reference, prediction

def Evalinfo(reference, prediction, language):
    """
    1. reference, prediction 각각 language에 맞는 전처리
    2. 
    """
    # 전처리
    if language == "korean":
        reference, prediction = replace_korean(reference, prediction)
    elif language == "english":
        reference, prediction = replace_english(reference, prediction)

    # Eval 값 load
    measures = compute_measures(reference, prediction)
    # truth에서 0번째 값만 load
    truth = measures['truth'][0]
    # hypothesis에서 0번째 값만 load
    hypothesis = measures['hypothesis'][0]
    # S,I,D,N,WER 값 load
    substitutions = measures['substitutions']
    insertions = measures['insertions']
    deletions = measures['deletions']
    total_words = len(reference.split())
    wer = get_eval(reference, prediction)
    return {
        "num": None,
        'reference': reference,
        'prediction': prediction,
        'truth': truth,
        'hypothesis': hypothesis,
        'S' : substitutions,
        'I' : insertions,
        'D' : deletions,
        'N' : total_words,
        'WER' : wer,
    }