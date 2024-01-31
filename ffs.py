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

from preprocess import preprocess_logits_for_metrics
from metrics import prepare_compute_metrics
from transformers import AutoModelForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, GenerationConfig, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from benchmarks.models import CodeGen
from tqdm import tqdm


class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=768, shuffle=False, load_range=None, include_buggy_line=True):
        self.data = []
        self.max_length = max_length

        fp = codecs.open(file_path, 'r', 'utf-8')
        for l in tqdm(fp.readlines()):
            l = eval(l)
            inputs = l['buggy function before']
            if include_buggy_line:
                inputs += l['buggy line']
            inputs += '<extra_id_0>' + l['buggy function after']
            outputs = l['fixed line'] + tokenizer.eos_token

            inputs = tokenizer.encode(inputs, return_tensors='pt')
            outputs = tokenizer.encode(outputs, return_tensors='pt')

            if inputs.shape[1] > max_length or outputs.shape[1] > max_length:
                continue

            self.data.append({
                'input_ids': inputs,
                'labels': outputs,
                'attention_mask': torch.ones(inputs.size()).long()
            })

            if len(self.data) % 10000 == 0:
                print('finish loading:', len(self.data))

            if load_range is not None and len(self.data) == load_range[1]:
                break

        if shuffle:
            random.seed(7)
            random.shuffle(self.data)

        print(file_path, 'total size:', len(self.data))
        if load_range is not None:
            self.data = self.data[load_range[0]: ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def data_collator(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_input_len = max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data

@click.command()
@click.option('--experiment_name', default='Experiment_1', type=str)
@click.option('--model_name', default='Salesforce/codet5-base', type=str)
@click.option('--dataset_name', default='CodeXGLUE-small', type=str)
@click.option('--checkpoint', default='Salesforce/codet5-base', type=str)
@click.option('--batch_size', default=4, type=int)
@click.option('--epochs', default=10, type=int)
@click.option('--max_length', default=768, type=int)
def train(experiment_name, model_name, dataset_name, checkpoint, batch_size, epochs, max_length):
    # Load Model TODO: Make file for it
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

    # Handle tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'Tokenizer: EOS: {tokenizer.eos_token} ({tokenizer(tokenizer.eos_token, truncation=True, padding=False, return_tensors="pt")}), PAD: {tokenizer.pad_token} ({tokenizer(tokenizer.pad_token, truncation=True, padding=False, return_tensors="pt")})')

    # Get Generation Config
    generation_config = GenerationConfig.from_model_config(model.generation_config)
    generation_config.max_new_tokens=512
    # generation_config.early_stopping=True
    # generation.num_beams=num_output
    # generation.num_return_sequences=num_output

    # Load dataset TODO: Make file for it
    training_file = '/home/machacini/codeRepair/datasets/clm/finetune_training.jsonl'
    validation_file = '/home/machacini/codeRepair/datasets/clm/finetune_validation.jsonl'
    train_dataset = CLMDataset(training_file, tokenizer, max_length=512, shuffle=False, load_range=None)
    eval_dataset = CLMDataset(validation_file, tokenizer, max_length=512, load_range=None)

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
    output_path = model_name
    output_path.replace("\\", "-")
    output_path = f"models/clm/codet5/{output_path}/"

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

    #print('Evaluating (Before)')
    #trainer.evaluate()
    print('Training')
    trainer.train()
    #print('Evaluating (After)')
    #trainer.evaluate()

if __name__ == '__main__':
    train()
