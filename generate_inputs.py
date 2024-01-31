import os
import sys
import json
import codecs
import subprocess
import time
import sys

from models import CodeGen, CodeT5, CodeGenInputConfig, CodeT5InputConfig, StarCoder, StarCoderInputConfig, DeepSeekCoder, DeepSeekCoderInputConfig, Bloom, BloomInputConfig, CodeLlama, CodeLlamaInputConfig


# Model Name, Benchmark name
if __name__ == '__main__':
    model_names = [sys.argv[1]]
    benchmark_name = sys.argv[2]
    BENCH_DIR = "/home/machacini/codeRepair/benchmarks/"
    JAVA_DIR = "/home/machacini/codeRepair/jasper"

    if benchmark_name == "humaneval":
        bench_dir = '/home/machacini/codeRepair/benchmarks/humaneval-java/'
    elif benchmark_name == "quixbugs":
        bench_dir = '/home/machacini/codeRepair/benchmarks/quixbugs/'
    elif benchmark_name == "defects4j":
        bench_dir = '/home/machacini/codeRepair/benchmarks/defects4j/'
    else:
        raise "Undefined benchmark"

    print(f"Running {benchmark_name} Benchmark")
    for model_name in model_names:
        # TODO: Use all models
        if model_name == "codegen":
            model = CodeGen(JAVA_DIR, BENCH_DIR)
            config = CodeGenInputConfig
        elif model_name == "codet5p":
            model = CodeT5(JAVA_DIR, BENCH_DIR)
            config = CodeT5InputConfig
        elif model_name == "starcoder":
            model = StarCoder(JAVA_DIR, BENCH_DIR)
            config = StarCoderInputConfig
        elif model_name == "deepseekcoder":
            model = DeepSeekCoder(JAVA_DIR, BENCH_DIR)
            config = DeepSeekCoderInputConfig
        elif model_name == "bloom":
            model = Bloom(JAVA_DIR, BENCH_DIR)
            config = BloomInputConfig
        elif model_name == "codellama":
            model = CodeLlama(JAVA_DIR, BENCH_DIR)
            config = CodeLlamaInputConfig
        else:
            raise f"Incorrent model name specified: {model_name}"

        for i, config in enumerate(config):
            if not os.path.exists(BENCH_DIR + f'results/{benchmark_name}/{model_name}'):
                os.makedirs(BENCH_DIR + f'results/{benchmark_name}/{model_name}')

            input_file = BENCH_DIR + f'results/{benchmark_name}/{model_name}/input_c' + str(i + 1) + '.json'
            print(f"==========Preparing input of {benchmark_name} benchmark to {model_name} model, Config: " + config + "==========")
            model.get_input(config, input_file, bench_dir=bench_dir)
            print("==========Input written to " + input_file)
