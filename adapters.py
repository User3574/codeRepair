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

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, IA3Config, PrefixTuningConfig, TaskType, PromptTuningInit, PromptTuningConfig, PeftType
from preprocess import preprocess_logits_for_metrics
from metrics import prepare_compute_metrics
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from benchmarks.models import CodeGen
from benchmarks.collators import seq2seq_collator
from benchmarks.models import models_classes, training_classes
from dataset import dataset_classes


@click.command()
@click.option('--experiment_name', default='Finetuning_CLM', type=str)
@click.option('--model_name', default='codet5p', type=str)
@click.option('--dataset_name', default='clm', type=str)
@click.option('--adapter_name', default='lora', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-small', type=str)
@click.option('--batch_size', default=4, type=int)
@click.option('--epochs', default=3, type=int)
@click.option('--max_length', default=768, type=int)
@click.option('--max_new_tokens', default=512, type=int)
def train(experiment_name, model_name, dataset_name, adapter_name, checkpoint, batch_size, epochs, max_length, max_new_tokens):
    # Get Model classes
    model_class = training_classes[model_name]
    tokenizer, model = model_class['tokenizer'], model_class['model']

    # Print
    print(f"Working with model {checkpoint}")

    # Load Model
    tokenizer = tokenizer.from_pretrained(checkpoint)
    model = model.from_pretrained(checkpoint, trust_remote_code=True).to("cuda")

    # Handle tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'Tokenizer: EOS: {tokenizer.eos_token} ({tokenizer(tokenizer.eos_token, truncation=True, padding=False, return_tensors="pt")}), PAD: {tokenizer.pad_token} ({tokenizer(tokenizer.pad_token, truncation=True, padding=False, return_tensors="pt")})')

    # Get Generation Config
    model.config.max_new_tokens = max_new_tokens

    # Load proper adapter
    if adapter_name == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    elif adapter_name == "iae3":
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            feedforward_modules=[]
        )
        model = get_peft_model(model, peft_config)

    elif adapter_name == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=20
        )
        model = get_peft_model(model, peft_config)

    elif adapter_name == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="Repair the following Java function:",
            tokenizer_name_or_path=model_name_or_path,
        )
        model = get_peft_model(model, peft_config)

    elif adapter_name == "none":
        print("Using no adapter")

    else:
        print("Invalid adapter")
        return_tensors

    # Load dataset
    dataset_class = dataset_classes[dataset_name]
    dataset, train_file, eval_file = dataset_class['dataset'], dataset_class['train_path'], dataset_class['eval_path']
    train_dataset = dataset(train_file, tokenizer, models_classes[model_name]['model'], max_length=max_length, shuffle=False, load_range=None)
    eval_dataset = dataset(eval_file, tokenizer, models_classes[model_name]['model'], max_length=max_length, load_range=None)

    # Load metrics
    hf_metrics = {
        "bleu": {"name": "bleu", "kwargs": {}},
        "rouge": {"name": "rouge", "kwargs": {}},
        "meteor": {"name": "meteor", "kwargs": {}},
        "exact_match": {"name": "exact_match", "kwargs": {}},
        "codebleu": {"name": "vichyt/metric-codebleu", "kwargs": {"lang": "java"}}
    }
    compute_metrics = prepare_compute_metrics(tokenizer, hf_metrics)

    # Set output path
    output_path = checkpoint
    output_path.replace("/", "-")
    output_path = f"adapters/{dataset_name}/{model_name}/{adapter_name}/{output_path}/"

    # Save tokenizer and initial model
    tokenizer.save_pretrained(output_path + "tokenizer/")
    model.save_pretrained(output_path + "checkpoint-0/")

    # Training settings
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=0,
        load_best_model_at_end=True,
        predict_with_generate=True,
        report_to="wandb",
        run_name=f"{dataset_name}_{checkpoint}_{adapter_name}",
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch'
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=seq2seq_collator,
        compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    print(f'Evaluating ({model_name} at {checkpoint})')
    print('Train dataset (Before)')
    results_train = trainer.evaluate(eval_dataset=train_dataset)
    print(results_train)
    print('Eval dataset (Before)')
    results_val = trainer.evaluate(eval_dataset=eval_dataset)
    print(results_val)

    print('Training')
    trainer.train()

    print(f'Evaluating ({model_name} at {checkpoint})')
    print('Train dataset (After)')
    results_train = trainer.evaluate(eval_dataset=train_dataset)
    print(results_train)
    print('Eval dataset (After)')
    results_val = trainer.evaluate(eval_dataset=eval_dataset)
    print(results_val)

if __name__ == '__main__':
    train()

