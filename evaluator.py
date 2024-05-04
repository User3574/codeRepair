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


@click.command()
@click.option('--experiment_name', default='Finetuning_CLM', type=str)
@click.option('--model_name', default='codet5p', type=str)
@click.option('--dataset_name', default='clm', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-small', type=str)
@click.option('--batch_size', default=4, type=int)
@click.option('--epochs', default=3, type=int)
@click.option('--max_length', default=768, type=int)
@click.option('--max_new_tokens', default=768, type=int)
@click.option('--evaluate_train', type=str)
def evaluator(experiment_name, model_name, dataset_name, checkpoint, batch_size, epochs, max_length, max_new_tokens, evaluate_train):
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
            checkpoint
        )
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

    # Get Generation Config
    model.config.max_new_tokens = max_new_tokens

    # Load dataset
    dataset_class = dataset_classes[dataset_name]
    dataset, train_file, eval_file = dataset_class['dataset'], dataset_class['train_path'], dataset_class['eval_path']
    train_dataset = dataset(train_file, tokenizer, models_classes[model_name]['model'], max_length=max_length, shuffle=False, load_range=None)
    eval_dataset = dataset(eval_file, tokenizer, models_classes[model_name]['model'], max_length=max_length, load_range=None)

    # Load metrics
    hf_metrics = {
        "exact_match": {"name": "exact_match", "kwargs": {}},
        "codebleu": {"name": "vichyt/metric-codebleu", "kwargs": {"lang": "java"}}
    }

    # Set output path
    output_path = checkpoint
    output_path.replace("/", "-")
    output_path = f"models/{dataset_name}/{model_name}/{output_path}/"

    # Load collators
    if task == "mask":
        collator = seq2seq_collator
        compute_metrics = prepare_compute_metrics(tokenizer, hf_metrics, False)
            
        # Training settings
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_path,
            do_train=True,
            do_eval=True,
            per_device_eval_batch_size=batch_size_test,
            num_train_epochs=epochs,
            seed=0,
            load_best_model_at_end=True,
            predict_with_generate=True,
            report_to="wandb",
            run_name=f"{dataset_name}_{checkpoint}_evaluate",
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch'
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
    else:
        collator = decoder_collator(tokenizer.pad_token_id)
        compute_metrics = prepare_compute_metrics(tokenizer, hf_metrics, True)
        
        # Training settings
        training_args = TrainingArguments(
            output_dir=output_path,
            do_train=True,
            do_eval=True,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            seed=0,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=f"{dataset_name}_{checkpoint}_evaluate",
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

    print(f'Evaluating ({model_name} at {checkpoint})')
    if evaluate_train == 'True':
        print('Train dataset (Before)')
        start = time.time()
        results_train = trainer.evaluate(eval_dataset=train_dataset)
        end = time.time()
        taken = end - start
        print(f'Time taken for Evaluating Train dataset: {taken}')
        print(results_train)
    
    print('Eval dataset (Before)')
    start = time.time()
    results_val = trainer.evaluate(eval_dataset=eval_dataset)
    end = time.time()
    taken = end - start
    print(f'Time taken for Evaluating Eval dataset: {taken}')
    print(results_val)

if __name__ == '__main__':
    evaluator()
