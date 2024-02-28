import os
import sqlite3
import json
import pickle
import pandas as pd
import torch
import codecs
import random

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, model_class, max_length=768, shuffle=False, load_range=None, include_buggy_line=True, training_task="promp"):
        self.data = []
        self.max_length = max_length

        fp = codecs.open(file_path, 'r', 'utf-8')
        for l in tqdm(fp.readlines()):
            l = eval(l)

            # prepare_input(self, fn_before, fn_bug, fn_fix, fn_after, eos_token)
            inputs, outputs = model_class.prepare_input(l['buggy function before'], l['buggy line'], l['fixed line'], l['buggy function after'], tokenizer.eos_token)
            
            if training_task is "prompt":
                prompt = inputs + '\n' + outputs
                prompt = tokenizer.encode(inputs, return_tensors='pt')
                
                if prompt.shape[1] > max_length:
                    continue
                
                self.data.append({
                    'input_ids': prompt,
                    'labels': prompt.clone(),
                    'attention_mask': torch.ones(prompt.size()).long()
                })
   
            elif training_task is "regressive":
                print(inputs, outputs)
                inputs = tokenizer.encode(inputs, return_tensors='pt')
                outputs = tokenizer.encode(outputs, return_tensors='pt')
                
                if inputs.shape[1] > max_length or outputs.shape[1] > max_length:
                    continue

                self.data.append({
                    'input_ids': inputs,
                    'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
                    'attention_mask': torch.ones(inputs.size()).long()
                })
            
            elif training_task is "mask":
                inputs = tokenizer.encode(inputs, return_tensors='pt')
                outputs = tokenizer.encode(outputs, return_tensors='pt')
                
                if inputs.shape[1] > max_length or outputs.shape[1] > max_length:
                    continue

                self.data.append({
                    'input_ids': inputs,
                    'labels': outputs,
                    'attention_mask': torch.ones(inputs.size()).long()
                })
                
            else:
                print('Training task not specified')
                raise

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


class CodeXGLUEDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, dataset_type, tokenizer, max_length=768, shuffle=False, load_range=None, include_buggy_line=True):
        self.data = []
        self.max_length = max_length

        buggy_path = f'{file_path}/{dataset_type}.buggy-fixed.buggy'
        fixed_path = f'{file_path}/{dataset_type}.buggy-fixed.fixed'

        buggy_file = open(buggy_path, "r")
        fixed_file = open(fixed_path, "r")

        # One function per row
        buggy = buggy_file.readlines()
        fixed = fixed_file.readlines()

        for i in tqdm(range(len(buggy))):
            # Get functions
            input_fn = buggy[i]
            output_fn = fixed[i]

            # Find Bug and Fix (Diff)
            info = {}
            start_line_idx, end_line_idx_input, end_line_idx_output = 0, -1, -1
            terminators = ['{', '}', ';']
            for j in range(min(len(input_fn), len(output_fn))):
                # New line has occured (Will be replaced)
                if input_fn[j] in terminators:
                    start_line_idx = j+1
                # Find start of error
                if (output_fn[j] != input_fn[j]):
                    break

            # Find line end for input_fn
            for k in range(start_line_idx+1, len(input_fn)):
                if input_fn[k] in terminators:
                    end_line_idx_input = k+1
                    break
            # Find line end for output_fn
            for k in range(start_line_idx+1, len(output_fn)):
                if output_fn[k] in terminators:
                    end_line_idx_output = k+1
                    break

            print(start_line_idx)
            print(f'Outpu: {output_fn}')
            print(f'Input: {input_fn}')

            info["buggy line"] = input_fn[start_line_idx:end_line_idx_input]
            info["fixed line"] = output_fn[start_line_idx:end_line_idx_output]

            # Split
            info["buggy function before"], info["buggy function after"] = input_fn[:start_line_idx], input_fn[end_line_idx_input:]

            if include_buggy_line:
                info["buggy function before"] += " // buggy line: " + info["buggy line"]

            # Create inputs, outputs
            inputs = info['buggy function before']
            inputs += '<extra_id_0>' + info['buggy function after']
            outputs = info['fixed line'] + tokenizer.eos_token

            # Tokenize
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


class CodeRep4Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, dataset_type, tokenizer, max_length=768, shuffle=False, load_range=None, include_buggy_line=True):
        self.data = []
        self.max_length = max_length

        buggy_path = f'{file_path}/src-{dataset_type}.txt'
        fixed_path = f'{file_path}/tgt-{dataset_type}.txt'

        buggy_file = open(buggy_path, "r")
        fixed_file = open(fixed_path, "r")

        # One function per row
        buggy = buggy_file.readlines()
        fixed = fixed_file.readlines()

        for i in tqdm(range(len(buggy))):
            # Get functions
            input_fn = buggy[i]
            output_fn = fixed[i]

            # Find Bug and Fix (Diff)
            info = {}
            info["buggy function before"], info["buggy function after"] = input_fn.split("<START_BUG>")
            info["buggy line"], info["buggy function after"] = info["buggy function after"].split("<END_BUG>")
            info["fixed line"] = output_fn

            print(info)

            if include_buggy_line:
                info["buggy function before"] += " // buggy line: " + info["buggy line"]

            print(info)

            # # Create inputs, outputs
            # inputs = info['buggy function before']
            # inputs += '<extra_id_0>' + info['buggy function after']
            # outputs = info['fixed line'] + tokenizer.eos_token

        #     # Tokenize
        #     inputs = tokenizer.encode(inputs, return_tensors='pt')
        #     outputs = tokenizer.encode(outputs, return_tensors='pt')
        #
        #     if inputs.shape[1] > max_length or outputs.shape[1] > max_length:
        #         continue
        #
        #     self.data.append({
        #         'input_ids': inputs,
        #         'labels': outputs,
        #         'attention_mask': torch.ones(inputs.size()).long()
        #     })
        #
        #     if len(self.data) % 10000 == 0:
        #         print('finish loading:', len(self.data))
        #
        #     if load_range is not None and len(self.data) == load_range[1]:
        #         break
        #
        # if shuffle:
        #     random.seed(7)
        #     random.shuffle(self.data)
        #
        # print(file_path, 'total size:', len(self.data))
        # if load_range is not None:
        #     self.data = self.data[load_range[0]: ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DeepFixDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, dataset_type, tokenizer, max_length=768, shuffle=False, load_range=None, include_buggy_line=True):
        self.data = []
        self.max_length = max_length

        buggy_path = f'{file_path}/src-{dataset_type}.txt'
        fixed_path = f'{file_path}/tgt-{dataset_type}.txt'

        buggy_file = open(buggy_path, "r")
        fixed_file = open(fixed_path, "r")

        # One function per row
        buggy = buggy_file.readlines()
        fixed = fixed_file.readlines()

        for i in tqdm(range(len(buggy))):
            # Get functions
            input_fn = buggy[i]
            output_fn = fixed[i]

            # Find Bug and Fix (Diff)
            info = {}
            info["buggy function before"], info["buggy function after"] = input_fn.split("<START_BUG>")
            info["buggy line"], info["buggy function after"] = info["buggy function after"].split("<END_BUG>")
            info["fixed line"] = output_fn

            print(info)

            if include_buggy_line:
                info["buggy function before"] += " // buggy line: " + info["buggy line"]

            print(info)

            # # Create inputs, outputs
            # inputs = info['buggy function before']
            # inputs += '<extra_id_0>' + info['buggy function after']
            # outputs = info['fixed line'] + tokenizer.eos_token

        #     # Tokenize
        #     inputs = tokenizer.encode(inputs, return_tensors='pt')
        #     outputs = tokenizer.encode(outputs, return_tensors='pt')
        #
        #     if inputs.shape[1] > max_length or outputs.shape[1] > max_length:
        #         continue
        #
        #     self.data.append({
        #         'input_ids': inputs,
        #         'labels': outputs,
        #         'attention_mask': torch.ones(inputs.size()).long()
        #     })
        #
        #     if len(self.data) % 10000 == 0:
        #         print('finish loading:', len(self.data))
        #
        #     if load_range is not None and len(self.data) == load_range[1]:
        #         break
        #
        # if shuffle:
        #     random.seed(7)
        #     random.shuffle(self.data)
        #
        # print(file_path, 'total size:', len(self.data))
        # if load_range is not None:
        #     self.data = self.data[load_range[0]: ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Models dictionary
dataset_classes = {}
dataset_classes['clm'] = {'dataset': CLMDataset, 'train_path': '/home/machacini/codeRepair/datasets/clm/finetune_training.jsonl', 'eval_path': '/home/machacini/codeRepair/datasets/clm/finetune_validation.jsonl'}

       
def create_deepfix(include_errors=False, db_path = 'datasets/DeepFix/dataset.db', valid_size=0.2, test_size=0.2, seed=0):
    data_dict = {'code_id': [], 'code': []}
    if include_errors:
        data_dict['error'] = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute('SELECT * FROM Code')
        names = list(map(lambda x: x[0], cursor.description))
        # print(names)

        cursor = conn.cursor()
        for row in cursor.execute("SELECT code_id, error, code FROM Code \n WHERE errorcount > 0"):
            code_id = str(row[0])
            error = row[1]
            code = row[2].encode('utf-8')

            data_dict['code_id'].append(code_id)
            data_dict['code'].append(code.decode("utf-8"))
            if include_errors:
                data_dict['error'].append(error)

    # Create dataframe
    df = pd.DataFrame.from_dict(data_dict)
    # Train, Test, Valid split
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    # Recalculate test_size
    valid_size = 1/(1-test_size) * valid_size
    train, valid = train_test_split(train, test_size=valid_size, random_state=seed)

    # Write to file
    with open('train.pkl', 'wb') as f:
       pickle.dump(train, f)

    with open('test.pkl', 'wb') as f:
       pickle.dump(test, f)

    with open('valid.pkl', 'wb') as f:
       pickle.dump(test, f)

    # Read
    # with open('pickled.pkl', 'rb') as f:
    #    loaded_dict = pickle.load(f)
    #    print(loaded_dict)

    return Dataset.from_dict(train), \
           Dataset.from_dict(test), \
           Dataset.from_dict(valid)


if __name__ == '__main__':
   #CodeXGLUEDataset('/home/machacini/codeRepair/datasets/CodeXGLUE/small', 'train', None, max_length=512)
   CodeRep4Dataset('/home/machacini/codeRepair/datasets/CodRep4', 'train', None, max_length=512)


