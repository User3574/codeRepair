import os
import sys
import json
import codecs
import subprocess
import time
import glob

from pathlib import Path
from models import models_classes


# Model Name, Benchmark name
if __name__ == '__main__':
    model_name = sys.argv[1]
    model_checkpoint = sys.argv[2]
    benchmark_name = sys.argv[3]
    output_name = sys.argv[4]
    adapter_name = sys.argv[5]
    
    BENCH_DIR = "/home/machacini/codeRepair/benchmarks/"
    JAVA_DIR = "/home/machacini/codeRepair/jasper"
    IS_FINETUNED = True

    if benchmark_name == "humaneval":
        bench_dir = '/home/machacini/codeRepair/benchmarks/results/humaneval/'
        max_new_tokens = 128
    elif benchmark_name == "quixbugs":
        bench_dir = '/home/machacini/codeRepair/benchmarks/results/quixbugs/'
        max_new_tokens = 128
    elif benchmark_name == "defects4j":
        bench_dir = '/home/machacini/codeRepair/benchmarks/results/defects4j/'
        max_new_tokens = 512
    else:
        raise "Undefined benchmark"

    # Get Model classes
    model_class = models_classes[model_name]['model']
    model = model_class(JAVA_DIR, BENCH_DIR, IS_FINETUNED)

    # Load Model, Tokenizer (is in Parent directory of all the checkpoints)
    root_checkpoint = str(Path(model_checkpoint).parent.absolute())
    tokenizer_checkpoint = root_checkpoint + "/tokenizer"
    print(f"Tokenizer: {tokenizer_checkpoint}, Model: {model_checkpoint}")

    # Iterate over Without, With Comments
    for input_file in glob.glob(f"{bench_dir}/{model_name}/finetuned/input_*"):
        print(f"Running {benchmark_name} Benchmark")
        print("==========Input from " + input_file)

        # Edit output name
        output_file = str(input_file).split('/')
        output_file[-1] = output_name + '_' + output_file[-1]
        output_file[-2] = output_file[-2] + f"/{adapter_name}"
        output_file = '/'.join(output_file)
        output_file = output_file.replace("input", "output")
        
        # Create folder if it doesn't exist
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"==========Generating output of {benchmark_name} benchmark to " + output_file + "==========")
        model.create_output(input_file, output_file, tokenizer_checkpoint, model_checkpoint, model_name, max_new_tokens)
        print("==========Output written to " + output_file)
