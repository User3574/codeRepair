import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, buggy, fixed):
        self.buggy = buggy
        self.fixed = fixed

    def __getitem__(self, idx):
        return {
            'attention_mask': self.buggy['attention_mask'][idx],
            'input_ids': self.buggy['input_ids'][idx],
            "labels": self.fixed['input_ids'][idx]
        }

    def __len__(self):
        return len(self.buggy["input_ids"])


def replace_unknown_labels(dataset, pad_token_id):
    labels = dataset["input_ids"].clone()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    dataset["labels"] = labels
    return dataset


def create_tokenized_dataset(tokenizer, prefix, dataset, max_length, replace_unknown):
    buggy = dataset['buggy']
    fixed = dataset['fixed']

    # Encode
    if prefix is not None:
        buggy = [prefix + code for code in buggy]
    buggy = tokenizer(
        buggy,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True
    )
    fixed = tokenizer(fixed, max_length=max_length, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")

    # Replace the index of the padding tokens by -100 (CrossEntropyLoss impact)
    pad_token_id = tokenizer.pad_token_id
    if replace_unknown is True:
        print('Replacing unknown labels')
        #buggy = replace_unknown_labels(buggy, pad_token_id)
        fixed = replace_unknown_labels(fixed, pad_token_id)

    return Dataset(buggy, fixed)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)
