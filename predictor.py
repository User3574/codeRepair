import torch
import os
import sqlite3
import json
import pickle
import pandas as pd
import torch
import click

from pathlib import Path
from preprocess import preprocess_logits_for_metrics
from metrics import prepare_compute_metrics
from benchmarks.models import CodeGen
from transformers import GenerationConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from benchmarks.collators import decoder_collator
from benchmarks.models import models_classes, training_classes
from dataset import dataset_classes


@click.command()
@click.option('--model_name', default='Salesforce/codet5-small', type=str)
@click.option('--dataset_name', default='CodeXGLUE-small', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-small', type=str)
@click.option('--batch_size', default=4, type=int)
@click.option('--max_length', default=768, type=int)
@click.option('--max_new_tokens', default=768, type=int)
def predict(model_name, dataset_name, checkpoint, batch_size, max_length, max_new_tokens):
    # Get Model classes
    model_class = training_classes[model_name]
    tokenizer, model, task = model_class['tokenizer'], model_class['model'], model_class['task']
    
    # Load Model, Tokenizer (is in Parent directory of all the checkpoints)
    model_checkpoint = checkpoint
    root_checkpoint = str(Path(model_checkpoint).parent.absolute())
    tokenizer_checkpoint = root_checkpoint + "/tokenizer"
    print(f"Tokenizer: {tokenizer_checkpoint}, Model: {model_checkpoint}")

    # Load Model
    tokenizer = tokenizer.from_pretrained(tokenizer_checkpoint, padding_side='left')
    model = model.from_pretrained(model_checkpoint, trust_remote_code=True).to("cuda")

    # Handle tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f'Tokenizer: EOS: {tokenizer.eos_token} ({tokenizer(tokenizer.eos_token, truncation=True, padding=False, return_tensors="pt")}), PAD: {tokenizer.pad_token} ({tokenizer(tokenizer.pad_token, truncation=True, padding=False, return_tensors="pt")})')

    # Get Generation Config
    model.config.max_new_tokens = max_new_tokens

    # Load dataset
    dataset_class = dataset_classes[dataset_name]
    dataset, train_file, eval_file = dataset_class['dataset'], dataset_class['train_path'], dataset_class['eval_path']
    train_dataset = dataset(train_file, tokenizer, models_classes[model_name]['model'], max_length=max_length, shuffle=False, load_range=100)
    eval_dataset = dataset(eval_file, tokenizer, models_classes[model_name]['model'], max_length=max_length, load_range=100)

    # Set output path
    output_path = f"{checkpoint}/evaluate"
    
    # Write to file
    data = data['train']
    dataloader = DataLoader(data, shuffle=True, batch_size=batch_size, collate_fn=decoder_collator)
    model.eval()
    for batch in dataloader:
        labels = batch['labels']
        outputs = model(**batch)
        loss = outputs.loss
        print(labels.shape)
        print(outputs.shape)


if __name__ == '__main__':
    predict()

