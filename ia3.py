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

from preprocess import preprocess_logits_for_metrics
from metrics import prepare_compute_metrics
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
from benchmarks.models import CodeGen
from benchmarks.collators import seq2seq_collator, decoder_collator
from benchmarks.models import models_classes, training_classes
from dataset import dataset_classes
from peft import LoraConfig, IA3Config, TaskType, get_peft_model


@click.command()
@click.option('--experiment_name', default='Finetuning_CLM', type=str)
@click.option('--model_name', default='codet5p', type=str)
@click.option('--dataset_name', default='clm', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-small', type=str)
@click.option('--batch_size_train', default=4, type=int)
@click.option('--batch_size_test', default=4, type=int)
@click.option('--epochs', default=3, type=int)
@click.option('--max_length', default=768, type=int)
@click.option('--max_new_tokens', default=768, type=int)
def train(experiment_name, model_name, dataset_name, checkpoint, batch_size_train, batch_size_test, epochs, max_length, max_new_tokens):
    # Get Model classes
    model_class = training_classes[model_name]
    tokenizer, model, task = model_class['tokenizer'], model_class['model'], model_class['task']
    
    # Print
    print(f"Working with model {checkpoint}")

    # Load Model
    tokenizer = tokenizer.from_pretrained(checkpoint, padding_side='left')
    model = model.from_pretrained(checkpoint, trust_remote_code=True).to("cuda")

    # Handle tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f'Tokenizer: EOS: {tokenizer.eos_token} ({tokenizer(tokenizer.eos_token, truncation=True, padding=False, return_tensors="pt")}), PAD: {tokenizer.pad_token} ({tokenizer(tokenizer.pad_token, truncation=True, padding=False, return_tensors="pt")})')

    # Get Generation Config
    model.config.max_new_tokens = max_new_tokens

    # Task type
    if task == "mask":
        task_type = TaskType.SEQ_2_SEQ_LM
    else:
        task_type = TaskType.CAUSAL_LM
    
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py
    if model_name == "codegen":
        target_modules = ["qkv_proj"]
        # Load proper adapter
        peft_config = IA3Config(
            task_type=task_type,
            inference_mode=False,
            target_modules=target_modules,
            feedforward_modules=[]
        )
    else:
        # Load proper adapter
        peft_config = IA3Config(
            task_type=task_type,
            inference_mode=False
        )
    model = get_peft_model(model, peft_config)

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
    output_path = f"models/ia3/{dataset_name}/{model_name}/{output_path}/"

    # Save tokenizer and initial model
    tokenizer.save_pretrained(output_path + "tokenizer/")
    model.save_pretrained(output_path + "checkpoint-0/")

    # Load collators
    if task == "mask":
        collator = seq2seq_collator
        compute_metrics = prepare_compute_metrics(tokenizer, hf_metrics, False)
        
        # Training settings
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_path,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=batch_size_train,
            per_device_eval_batch_size=batch_size_test,
            num_train_epochs=epochs,
            seed=0,
            load_best_model_at_end=True,
            predict_with_generate=True,
            report_to="wandb",
            run_name=f"{dataset_name}_{checkpoint}",
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
            per_device_train_batch_size=batch_size_train,
            per_device_eval_batch_size=batch_size_test,
            num_train_epochs=epochs,
            seed=0,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=f"ia3_{dataset_name}_{checkpoint}",
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

    print('Training')
    start = time.time()
    trainer.train()
    end = time.time()
    taken = end - start
    print(f'Time taken for Training: {taken}')

    print(f'Evaluating ({model_name} at {checkpoint})')
    print('Train dataset (After)')
    results_train = trainer.evaluate(eval_dataset=train_dataset)
    print(results_train)
    print('Eval dataset (After)')
    results_val = trainer.evaluate(eval_dataset=eval_dataset)
    print(results_val)

if __name__ == '__main__':
    train()

