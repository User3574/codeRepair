import torch
import os
import sqlite3
import json
import pickle
import pandas as pd
import torch
import codecs
import random
import click
import time

from peft import PeftModel
from pathlib import Path
from preprocess import preprocess_logits_for_metrics
from metrics import prepare_compute_metrics
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
from benchmarks.models import CodeGen
from benchmarks.collators import seq2seq_collator, decoder_collator
from benchmarks.models import models_classes, training_classes
from dataset import dataset_classes


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


@click.command()
@click.option('--model_name', default='codet5p', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-small', type=str)
def analyse(model_name, checkpoint):
    # Get Model classes
    model_class = training_classes[model_name]
    tokenizer, model, task = model_class['tokenizer'], model_class['model'], model_class['task']
    
    # Print
    print(f"Working with model {checkpoint}")
    root_path = Path(checkpoint)
    tokenizer_path = str(root_path.parent.absolute()) + "/tokenizer"

    # Load Model
    tokenizer = tokenizer.from_pretrained(tokenizer_path, padding_side='left')
    model = model.from_pretrained(checkpoint, trust_remote_code=True).to("cuda")
    
    # Try to load PEFT
    try:
        model = PeftModel.from_pretrained(
            model,
            checkpoint,
            is_trainable=True
        )
        print_trainable_parameters(model)
        model = model.merge_and_unload()
        print('PEFT model loaded successfully')
    except:
        print('Model loaded')
    
    # Handle tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f'Tokenizer: EOS: {tokenizer.eos_token} ({tokenizer(tokenizer.eos_token, truncation=True, padding=False, return_tensors="pt")}), PAD: {tokenizer.pad_token} ({tokenizer(tokenizer.pad_token, truncation=True, padding=False, return_tensors="pt")})')

    #print_trainable_parameters(model)

if __name__ == '__main__':
    analyse()


