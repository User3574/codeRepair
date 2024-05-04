import sys
import json
import os


def prepare_input(fn_before, fn_bug, fn_after, model_name):
    if model_name == 'CODEGEN':
        inputs = fn_before + "// bug start: \n" + fn_bug + "// bug end \n" + fn_after + "// fix: \n"
    elif model_name == 'CODET5':
        inputs = fn_before + "// bug start: \n" + fn_bug + "// bug end \n" + fn_after
    elif model_name == 'STARCODER':
        fn_bug = "// bug start: \n" + fn_bug + "// bug end \n"
        inputs = "<fim_prefix>" + fn_before + fn_bug + "<fim_suffix>\n" + fn_after + "<fim_middle>"
    elif model_name == 'DEEPSEEKCODER':
        fn_bug = "// bug start: \n" + fn_bug + "// bug end \n"
        inputs = "<｜fim▁begin｜>" + fn_before + fn_bug + "<｜fim▁hole｜>\n" + fn_after + "<｜fim▁end｜>"
    elif model_name == "BLOOM":
        inputs = fn_before + "// bug start: \n" + fn_bug + "// bug end \n" + fn_after + "// fix: \n"
    elif model_name == "CODELLAMA":
        fn_bug = "// bug start: \n" + fn_bug + "// bug end \n"
        inputs = fn_before + fn_bug + "<FILL_ME>\n" + fn_after
    else:
        raise f'Invalid model name: {model_name}'
    return inputs


if __name__ == '__main__':
    path = sys.argv[1]
    model_names = ['CODEGEN', 'CODET5', 'STARCODER', 'DEEPSEEKCODER', 'BLOOM', 'CODELLAMA']

    for model_name in model_names:
        result = json.load(open(path, 'r'))
        result['config'] = model_name + '_FINETUNED'
        for task in result['data']:
            task_input = result['data'][task]['input']

            # Remove line containing <extra_id_0>
            task_lines = []
            for input_line in task_input.split('\n'):
                if "<extra_id_0>" not in input_line:
                    task_lines.append(input_line)

            # Split into fn_before, fn_bug, fn_after
            bug_start, bug_length = None, 0
            for idx, input_line in enumerate(task_lines):
                if '// buggy line:' in input_line:
                    if bug_start is None:
                        bug_start = idx
                    bug_length += 1

            # Keep tests that were not compiled
            if bug_start is None:
                continue

            # Split original task_input
            fn_before = task_lines[:bug_start]
            fn_bug = task_lines[bug_start:bug_start+bug_length]
            fn_after = task_lines[bug_start+bug_length:]

            # Postprocessing of buggy lines
            for idx, buggy_line in enumerate(fn_bug):
                fn_bug[idx] = buggy_line.replace('// buggy line:', '')

            # Join
            fn_before = '\n'.join(fn_before) + '\n'
            fn_bug = '\n'.join(fn_bug) + '\n'
            fn_after = '\n'.join(fn_after)
            # Apply
            result['data'][task]['input'] = prepare_input(fn_before, fn_bug, fn_after, model_name)

        save_path = os.path.dirname(os.path.abspath(path)) + f'/input_{model_name}_finetuned.json'
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
