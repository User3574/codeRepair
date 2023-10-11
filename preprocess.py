import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, buggy, fixed):
        self.buggy = buggy
        self.fixed = fixed

    def __getitem__(self, idx):
        return {
            'attention_mask': self.buggy['attention_mask'][idx],
            'token_type_ids': self.buggy['token_type_ids'][idx],
            'input_ids': self.buggy['input_ids'][idx],
            "decoder_input_ids": self.fixed['input_ids'][idx],
            "labels": self.fixed['input_ids'][idx]
        }

    def __len__(self):
        return len(self.buggy["input_ids"])


def replace_unknown(labels):
    fixes = []
    for tokenized in labels:
        fixed = [token if token != 0 else -100 for token in tokenized]
        fixes.append(fixed)
    return fixes


def create_tokenized_dataset(tokenizer, prefix, dataset, max_length):
    buggy = dataset['buggy']
    fixed = dataset['fixed']

    # Encode
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
    fixed = tokenizer(fixed, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    # Replace the index of the padding tokens by -100 (CrossEntropyLoss impact)
    # buggy['input_ids'] = replace_unknown(buggy['input_ids'])
    # fixed['input_ids'] = replace_unknown(fixed['input_ids'])

    return Dataset(buggy, fixed)
