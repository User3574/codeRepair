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
from benchmarks.models import CodeGen, models_classes
from transformers import GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from benchmarks.collators import data_collator
from dataset import dataset_classes


@click.command()
@click.option('--experiment_name', default='Experiment_1', type=str)
@click.option('--model_name', default='Salesforce/codet5-small', type=str)
@click.option('--dataset_name', default='CodeXGLUE-small', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-small', type=str)
@click.option('--batch_size', default=1, type=int)
@click.option('--max_length', default=768, type=int)
def train(experiment_name, model_name, dataset_name, checkpoint, batch_size, max_length):
    # Get Model classes
    model_class = models_classes[model_name]
    tokenizer, model = model_class['tokenizer'], model_class['model']

    # Load Model, Tokenizer (is in Parent directory of all the checkpoints)
    root_checkpoint = str(Path(checkpoint).parent.absolute())
    tokenizer = tokenizer.from_pretrained(root_checkpoint + "/tokenizer")
    model = model.from_pretrained(checkpoint, trust_remote_code=True).to("cuda")

    # Handle padding tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'Tokenizer: EOS: {tokenizer.eos_token} ({tokenizer(tokenizer.eos_token, truncation=True, padding=False, return_tensors="pt")}), PAD: {tokenizer.pad_token} ({tokenizer(tokenizer.pad_token, truncation=True, padding=False, return_tensors="pt")})')

    # Get Generation Config
    generation_config = GenerationConfig.from_model_config(model.generation_config)
    generation_config.max_new_tokens=512
    # generation_config.early_stopping=True
    # generation.num_beams=num_output
    # generation.num_return_sequences=num_output

    # Load dataset
    dataset_class = dataset_classes[dataset_name]
    dataset, train_file, eval_file = dataset_class['dataset'], dataset_class['train_path'], dataset_class['eval_path']
    train_dataset = dataset(train_file, tokenizer, max_length=512, shuffle=False, load_range=None)
    eval_dataset = dataset(eval_file, tokenizer, max_length=512, load_range=None)

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
    output_path = f"{checkpoint}/evaluate"

    # Training settings
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        do_train=False,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=0,
        load_best_model_at_end=True,
        predict_with_generate=True,
        report_to="wandb",
        run_name="train",
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        generation_config=generation_config
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    print(f'Evaluating ({model_name} at {checkpoint})')
    print('Train dataset')
    trainer.evaluate(eval_dataset=train_dataset)
    print('Eval dataset')
    trainer.evaluate(eval_dataset=eval_dataset)

if __name__ == '__main__':
    train()

