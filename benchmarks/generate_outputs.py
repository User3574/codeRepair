import os
import sys
import json
import codecs
import subprocess
import time

from models import CodeGen, CodeT5, CodeGenInputConfig, CodeT5InputConfig, StarCoder, StarCoderInputConfig, DeepSeekCoder, DeepSeekCoderInputConfig, Bloom, BloomInputConfig, CodeLlama, CodeLlamaInputConfig


# Model Name, Benchmark name
if __name__ == '__main__':
    model_names = [sys.argv[1]]
    benchmark_name = sys.argv[2]
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

    print(f"Running {benchmark_name} Benchmark")
    for model_name in model_names:
        if model_name == "codegen":
            model = CodeGen(JAVA_DIR, BENCH_DIR)
            checkpoint_names = ["Salesforce/codegen-6B-multi"]#["Salesforce/codegen-350M-multi", "Salesforce/codegen-2B-multi", "Salesforce/codegen-6B-multi"]
            config = CodeGenInputConfig
        elif model_name == "codet5p":
            model = CodeT5(JAVA_DIR, BENCH_DIR)
            checkpoint_names = ["Salesforce/codet5-small", "Salesforce/codet5-base", "Salesforce/codet5-large"]
            config = CodeT5InputConfig
        elif model_name == "starcoder":
            model = StarCoder(JAVA_DIR, BENCH_DIR)
            checkpoint_names = ["bigcode/starcoderbase-3b", "bigcode/starcoderbase-7b"]#["bigcode/starcoderbase-1b", "bigcode/starcoderbase-3b", "bigcode/starcoderbase-7b"]
            config = StarCoderInputConfig
        elif model_name == "deepseekcoder":
            model = DeepSeekCoder(JAVA_DIR, BENCH_DIR)
            checkpoint_names = ["deepseek-ai/deepseek-coder-6.7b-base"]#["deepseek-ai/deepseek-coder-1.3b-base", "deepseek-ai/deepseek-coder-6.7b-base"]
            config = DeepSeekCoderInputConfig
        elif model_name == "bloom":
            model = Bloom(JAVA_DIR, BENCH_DIR)
            checkpoint_names = ["bigscience/bloom-560m", "bigscience/bloom-1b7", "bigscience/bloom-7b1"]
            config = BloomInputConfig
        elif model_name == "codellama":
            model = CodeLlama(JAVA_DIR, BENCH_DIR)
            checkpoint_names = ["codellama/CodeLlama-7b-hf"]
            config = CodeLlamaInputConfig
        else:
            raise f"Incorrent model name specified: {model_name}"

        for i, config in enumerate(config):
            input_file = BENCH_DIR + f'results/{benchmark_name}/{model_name}/input_c' + str(i + 1) + '.json'
            print("==========Input from " + input_file)

            # Iterate over all checkpoints
            for checkpoint_name in checkpoint_names:
                output_file = BENCH_DIR + f'results/{benchmark_name}/{model_name}/' + '_'.join(
                    checkpoint_name.replace('/', '-').split('-')) + '_output_c' + str(i + 1) + '.json'
                print(
                    f"==========Generating output of {benchmark_name} benchmark by " + checkpoint_name + ", Config: " + config + "==========")
                model.create_output(input_file, output_file, checkpoint_name, checkpoint_name, checkpoint_name, max_new_tokens)
                print("==========Output written to " + output_file)
