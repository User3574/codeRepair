import glob
import sys
import time

from benchmark import HumanEval, QuixBugs, Defects4j


if __name__ == '__main__':
    model_names = [sys.argv[1]]
    benchmark_name = sys.argv[2]
    relative_path = sys.argv[3]

    result_folder = "/home/machacini/codeRepair/benchmarks/results/" + benchmark_name
    bench_dir = "/home/machacini/codeRepair/benchmarks"

    for model_name in model_names:
        print(f"Validating Benchmark {benchmark_name}")
        model_outputs = f"{result_folder}/{model_name}/"
        model_outputs += f"{relative_path}"
        model_outputs = glob.glob(f"{model_outputs}/*output*")
        print("Validating: ", model_outputs)

        for i, model_output in enumerate(model_outputs):
            print(f"{model_output}, {i}")
            if benchmark_name == "humaneval":
                bench = HumanEval(f"{bench_dir}/humaneval-java/", f"{bench_dir}/tmp/humaneval-java/{model_name}/{i}/")
            elif benchmark_name == "quixbugs":
                bench = QuixBugs(f"{bench_dir}/quixbugs/", f"{bench_dir}/tmp/quixbugs/{model_name}/{i}/")
            elif benchmark_name == "defects4j":
                bench = Defects4j(f"{bench_dir}/defects4j/", f"{bench_dir}/tmp/defects4j/{model_name}/{i}/")
            else:
                print("Invalid bench name: {benchmark_name}")

            valid_file = model_output.replace("output", "valid")
            print(f"Validating: {model_output} to {valid_file}")
            bench.validate(model_output, valid_file)
