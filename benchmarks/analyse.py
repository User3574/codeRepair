import sys
import json
import glob
import os
import pandas as pd

from collections import Counter

# Model Name, Benchmark name
if __name__ == '__main__':
    model_path = sys.argv[1]
    benchmark_name = sys.argv[2]
    model_name = sys.argv[3]

    results = {}
    counters = {}

    for filepath in glob.glob(str(model_path) + '/*_valid_*.json'):
        name = os.path.basename(filepath)
        f = open(filepath)
        data = json.load(f)["data"]
        correctness = {}
        for project, info in data.items():
            correctness[str(project)] = {"uncompilable": 0, "plausible": 0, "timeout": 0, "wrong": 0}
            for patch in info["output"]:
                correctness[str(project)][str(patch["correctness"])] += 1
        f.close()

        table = pd.DataFrame.from_dict(correctness).transpose()

        # Add Result
        result = []
        for index, row in table.iterrows():
            if row["plausible"] > 0:
                result.append("compilable")
            elif row["wrong"] > 0:
                result.append("wrong")
            elif row["uncompilable"] > 0:
                result.append("uncompilable")
            elif row["timeout"] > 0:
                result.append("timeout")
            else:
                result.append("unknown")
        table = table.assign(result=result)

        latex_table = table.to_latex(
            index=False,
            caption=f"{name}",
            position="htbp",
            column_format="|l|l|l|l|",
            escape=False
        )
        # print(latex_table)
        counters[name] = dict(sorted(Counter(result).items()))

    counters = pd.DataFrame.from_dict(counters).transpose()
    counters = pd.DataFrame.fillna(counters, 0)
    counters = counters.reindex(sorted(counters.columns), axis=1)
    for i in counters.columns:
        counters[[i]] = counters[[i]].astype(float).astype(int)

    counters_latex = counters.to_latex(
        index=True,
        caption=f"Results of the {model_name} for {benchmark_name}",
        position="htbp",
        column_format="lccccc",
        escape=False
    )
    print(counters_latex)
