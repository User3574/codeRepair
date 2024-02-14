import torch
import click
import evaluate

from tqdm import tqdm
from preprocess import create_tokenized_dataset, preprocess_logits_for_metrics
from transformers import AutoTokenizer
from models import get_model
from dataset import get_dataset
from trl import PPOTrainer, PPOConfig, DPOTrainer, DPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch


@click.command()
@click.option('--experiment_name', default='Experiment_1', type=str)
@click.option('--model_name', default='CodeT5', type=str)
@click.option('--dataset_name', default='CodeXGLUE-small', type=str)
@click.option('--checkpoint', default='Salesforce/codet5p-220m-py', type=str)
@click.option('--batch_size', default=128, type=int)
@click.option('--epochs', default=5, type=int)
@click.option('--max_length', default=256, type=int)
@click.option('--replace_unknown', default=True)
@click.option("--task_prefix", default=None)
@click.option("--algorithm", default="DPO")
def pretrain(experiment_name, model_name, dataset_name, checkpoint, batch_size, epochs, max_length, replace_unknown, task_prefix, algorithm):
    # Get Models
    model, tokenizer = get_model(model_name, checkpoint)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model, device_map='auto', return_dict=True)
    model_ref = create_reference_model(model)
    
    # TODO: Is this required?
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print('Getting dataset')
    train, test, valid = get_dataset(dataset_name)
    train_dataset = create_tokenized_dataset(tokenizer, task_prefix, train, max_length, replace_unknown)

    # Setting up our Reward model = BLEU metric, CODEBLEU, Number of Passed test(s), Human value
    # Google BLEU is used for sentence-sentece comparison
    # TODO: Composite rewards, models?
    metric = "google_bleu"
    reward_system = evaluate.load(metric)

    if algorithm == "PPO":
        ppo_config = PPOConfig(
            batch_size=batch_size,
            ppo_epochs=epochs,
            mini_batch_size=128,
            gradient_accumulation_steps=1,
            learning_rate=1.41e-5
        )
        trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, train_dataset)
    elif algorithm == "DPO":
        # TODO: Train https://huggingface.co/blog/dpo-trl
        dpo_config = DPOConfig(
            batch_size=batch_size,
            ppo_epochs=epochs,
            mini_batch_size=128,
            gradient_accumulation_steps=1,
            learning_rate=1.41e-5
        )
        trainer = DPOTrainer(dpo_config, model, model_ref, tokenizer, train_dataset)

    # Arguments for generator
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        # what if we keep padding token
        "pad_token_id": tokenizer.pad_token_id,
        # max_length is required (20 default)
        "max_new_tokens": max_length
    }
    
    
    # Training loop
    print('Training')
    for epoch, batch in tqdm(enumerate(trainer.dataloader)):
        query_tensors = list(batch["input_ids"])
        
        # Get responses
        response_tensors = []
        for query_tensor in query_tensors:
            response_tensor = trainer.generate(query_tensor, **generation_kwargs)
            response_tensors.append(response_tensor.squeeze()[-max_length:])
            
        # Translate to text from predictions, labels
        batch["response"] = [tokenizer.decode(response.squeeze(), skip_special_tokens=True) for response in response_tensors]
        batch["target"] = [tokenizer.decode(label, skip_special_tokens=True) for label in batch["labels"]]
        L = len(batch["response"])
        
        # Calculate rewards
        rewards = [torch.tensor(reward_system.compute(predictions=[batch["response"][i]], references=[[batch["target"][i]]])[metric]) for i in range(L)]

        # Run PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards)


if __name__ == '__main__':
    pretrain()
