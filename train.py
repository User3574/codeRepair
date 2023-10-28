import click
import os
from preprocess import create_tokenized_dataset, preprocess_logits_for_metrics
from metrics import prepare_compute_metrics
from models import get_model
from dataset import get_dataset
from transformers import TrainingArguments, Trainer


@click.command()
@click.option('--experiment_name', default='Experiment_1', type=str)
@click.option('--model_name', default='CodeT5', type=str)
@click.option('--dataset_name', default='CodeXGLUE-small', type=str)
@click.option('--checkpoint', default='Salesforce/codet5p-220m-py', type=str)
@click.option('--batch_size', default=32, type=int)
@click.option('--epochs', default=5, type=int)
@click.option('--max_length', default=512, type=int)
@click.option('--replace_unknown', default=False) # If switched to True it causes crash, TODO: FIX
@click.option('--do_test', default=True)
def train(experiment_name, model_name, dataset_name, checkpoint, batch_size, epochs, max_length, replace_unknown, do_test):
    # Set wandb vars
    os.environ["WANDB_PROJECT"]=experiment_name + model_name + dataset_name
    os.environ["WANDB_LOG_MODEL"]="checkpoint"

    # GET_MODEL
    model, tokenizer = get_model(model_name, checkpoint)
    
    # GET METRICS
    hf_metrics = {
            "bleu": {"name": "bleu", "kwargs": {}},
            #"rouge": {"name": "rouge", "kwargs": {}},
            #"meteor": {"name": "meteor", "kwargs": {}},
            #"codebleu": {"name": "vichyt/metric-codebleu", "kwargs": {"lang": "java"}}
    }
    compute_metrics = prepare_compute_metrics(tokenizer, hf_metrics)

    # GET DATASET
    train, test, valid = get_dataset(dataset_name)

    # APPLY TOKENIZER
    task_prefix = "Repair Java function: "
    train_dataset = create_tokenized_dataset(tokenizer, task_prefix, train, max_length, replace_unknown)
    valid_dataset = create_tokenized_dataset(tokenizer, task_prefix, valid, max_length, replace_unknown)
    print(len(train_dataset))

    # Included wandb: https://docs.wandb.ai/guides/integrations/huggingface
    # Trainer Arguments
    # args = TrainingArguments(
    #    output_dir="output",
    #    evaluation_strategy="epoch",
    #    save_strategy="epoch",
    #    do_train=True,
    #    do_eval=True,
    #    per_device_train_batch_size=batch_size,
    #    per_device_eval_batch_size=batch_size,
    #    num_train_epochs=epochs,
    #    seed=0,
    #    load_best_model_at_end=True,
    #)

    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        report_to="wandb",
        run_name="train",
        seed=0,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Train pre-trained model
    fine_tuned_model_path = f'models/{dataset_name}/{model_name}/{checkpoint}'
    # Evaluate raw model
    trainer.evaluate()
    # Fine-tune the raw model
    #trainer.train()
    trainer.save_model(fine_tuned_model_path)
    
    if do_test:
        # ------------------------------------ Testing -----------------------
        # GET_MODEL (Already fine-tuned)
        model, tokenizer = get_model(model_name, fine_tuned_model_path)

        args = TrainingArguments(
            output_dir="output",
            do_train=False,
            do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to="wandb",
            run_name="test",
            seed=0,
        )

        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        trainer.evaluate()

if __name__ == '__main__':
    train()

