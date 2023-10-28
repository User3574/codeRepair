import os
import sqlite3
import json
import pickle
import pandas as pd

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name):
    if dataset_name == "CodeXGLUE-small":
        train, test, valid = load_codexglue("small")
    elif dataset_name == "CodeXGLUE-medium":
        train, test, valid = load_codexglue("medium")
    elif dataset_name == "DeepFix":
        train, test, valid = load_deepfix()
    elif dataset_name == "CodeRep4":
        train, test, valid = load_codrep()
    elif dataset_name == "":
        train, test, valid = load_fixeval()
    return train, test, valid
        

def load_fixeval(language="python", datapath="./datasets/FixEval"):
    datasets = {'train': {'src' : f'src_train.{language}-{language}.{language}', 'tgt': f'tgt_train.{language}-{language}.{language}'}, 
                'valid': {'src': f'src_valid.{language}-{language}.{language}', 'tgt': f'tgt_valid.{language}-{language}.{language}'}, 
                'test': {'src': f'src_test.{language}-{language}.{language}', 'tgt': f'tgt_test.{language}-{language}.{language}'}}
    datapath = os.path.join(datapath, language, 'processed')

    for k in datasets:
        src_file, tgt_file = datasets[k]['src'], datasets[k]['tgt']
        buggy_path = os.path.join(datapath, src_file)
        fixed_path = os.path.join(datapath, tgt_file)

        buggy_file = open(buggy_path, "r")
        fixed_file = open(fixed_path, "r")

        # One function per row
        buggy_functions = buggy_file.readlines()
        fixed_functions = fixed_file.readlines()

        datasets[k] = {'buggy': buggy_functions, 'fixed': fixed_functions}

    return Dataset.from_dict(datasets['train']), \
           Dataset.from_dict(datasets['test']), \
           Dataset.from_dict(datasets['valid'])

def load_codrep(datapath="./datasets/CodRep4"):
    datasets = {'train': {'src' : 'src-train.txt', 'tgt': 'tgt-train.txt'}, 
                'valid': {'src': 'src-val.txt', 'tgt': 'tgt-val.txt'}, 
                'test': {'src': 'src-test.txt', 'tgt': 'tgt-test.txt'}}
    
    for k in datasets:
        src_file, tgt_file = datasets[k]['src'], datasets[k]['tgt']
        buggy_path = os.path.join(datapath, src_file)
        fixed_path = os.path.join(datapath, tgt_file)

        buggy_file = open(buggy_path, "r")
        fixed_file = open(fixed_path, "r")

        # One function per row
        buggy_functions = buggy_file.readlines()
        fixed_functions = fixed_file.readlines()

        datasets[k] = {'buggy': buggy_functions, 'fixed': fixed_functions}

    return Dataset.from_dict(datasets['train']), \
           Dataset.from_dict(datasets['test']), \
           Dataset.from_dict(datasets['valid'])
       
def load_codexglue(dataset="small", datapath="./datasets/CodeXGLUE"):
    path = os.path.join(datapath, dataset)
    buggy_suffix = '.buggy-fixed.buggy'
    fixed_suffix = '.buggy-fixed.fixed'
    datasets = {'train': None, 'test': None, 'valid': None}

    for k in datasets.keys():
        buggy_path = os.path.join(path, k + buggy_suffix)
        fixed_path = os.path.join(path, k + fixed_suffix)

        buggy_file = open(buggy_path, "r")
        fixed_file = open(fixed_path, "r")

        # One function per row
        buggy_functions = buggy_file.readlines()
        fixed_functions = fixed_file.readlines()

        datasets[k] = {'buggy': buggy_functions, 'fixed': fixed_functions}

    return Dataset.from_dict(datasets['train']), \
           Dataset.from_dict(datasets['test']), \
           Dataset.from_dict(datasets['valid'])
       
       
def load_deepfix(include_errors=False, db_path = 'datasets/DeepFix/dataset.db', valid_size=0.2, test_size=0.2, seed=0):
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
    # with open('pickled.pkl', 'wb') as f:
    #    pickle.dump(data_dict, f)

    # Read
    # with open('pickled.pkl', 'rb') as f:
    #    loaded_dict = pickle.load(f)
    #    print(loaded_dict)
    
    return Dataset.from_dict(train), \
           Dataset.from_dict(test), \
           Dataset.from_dict(valid)

#if __name__ == '__main__':
#    train, test, valid = load_fixeval()
#    print(train, test, valid)
