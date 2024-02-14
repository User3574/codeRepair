import os
import sys
import json
import codecs
import subprocess
import time

from pathlib import Path
from models import models_classes


# Model Name, Benchmark name
if __name__ == '__main__':
    model_name = sys.argv[1]
    benchmark_name = sys.argv[2]
    model_checkpoint = sys.argv[3]
    BENCH_DIR = "/home/machacini/codeRepair/benchmarks/"
    JAVA_DIR = "/home/machacini/codeRepair/jasper"

    if benchmark_name == "humaneval":
        bench_dir = '/home/machacini/codeRepair/benchmarks/humaneval-java/'
        max_new_tokens = 128
    elif benchmark_name == "quixbugs":
        bench_dir = '/home/machacini/codeRepair/benchmarks/quixbugs/'
        max_new_tokens = 128
    elif benchmark_name == "defects4j":
        bench_dir = '/home/machacini/codeRepair/benchmarks/defects4j/'
        max_new_tokens = 512
    else:
        raise "Undefined benchmark"

    # Get Model classes
    model_class = models_classes[model_name]['model']
    model = model_class(JAVA_DIR, BENCH_DIR)

    # Load Model, Tokenizer (is in Parent directory of all the checkpoints)
    root_checkpoint = str(Path(model_checkpoint).parent.absolute())
    tokenizer_checkpoint = root_checkpoint + "/tokenizer"
    print(f"Tokenizer: {tokenizer_checkpoint}, Model: {model_checkpoint}")

    # Iterate over Without, With Comments
    for i in range(2):
        print(f"Running {benchmark_name} Benchmark")
        input_file = BENCH_DIR + f'results/{benchmark_name}/{model_name}/input_c' + str(i + 1) + '.json'
        print("==========Input from " + input_file)

        # Iterate over all checkpoints
        output_file = BENCH_DIR + f'results/{benchmark_name}/{model_name}/finetuned/output_c' + str(i + 1) + '.json'
        print(f"==========Generating output of {benchmark_name} benchmark to " + output_file + "==========")
        model.create_output(input_file, output_file, tokenizer_checkpoint, model_checkpoint, model_name, max_new_tokens)
        print("==========Output written to " + output_file)
