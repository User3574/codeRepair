import os
from datasets import load_dataset, Dataset


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


if __name__ == '__main__':
    train, test, valid = load_codexglue("small")
    for d in train:
        print(d)
