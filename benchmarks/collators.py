import torch

def seq2seq_collator(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_input_len = max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    
    # Pad to the right
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
    
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data

def decoder_collator(batch, pad_token_id=50256):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_input_len = max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    
    # Pad to the left (input size is same as output size)
    for b in batch:
        batch_data['input_ids'].append(torch.cat([torch.zeros(1, max_input_len - b['input_ids'].size(1)).fill_(pad_token_id).long(), b['input_ids']], dim=1))
        batch_data['labels'].append(torch.cat([torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long(), b['labels']], dim=1))
        batch_data['attention_mask'].append(torch.cat([torch.zeros(1, max_input_len - b['attention_mask'].size(1)), b['attention_mask']], dim=1))
        
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data

