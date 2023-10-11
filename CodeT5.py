from transformers import T5ForConditionalGeneration, AutoTokenizer, RobertaTokenizer, TrainingArguments, Trainer
from data import load_codexglue
from preprocess import create_tokenized_dataset
from metrics import compute_metrics

"""
On hugging face site is model_name + info (tokenizer, model)
"""

if __name__ == '__main__':
    # Settings
    device = "cuda"
    checkpoint = "Salesforce/codet5p-220m-py"
    batch_size=1
    epochs=5

    # Load pretrained model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

    # Load dataset
    train, test, valid = load_codexglue("small")
    task_prefix = "Repair Java function: "
    example = task_prefix + train['buggy'][0]
    train_dataset = create_tokenized_dataset(tokenizer, task_prefix, train, 64)
    valid_dataset = create_tokenized_dataset(tokenizer, task_prefix, valid, 64)

    # Trainer Arguments
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=0,
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # Train pre-trained model
    trainer.train()
