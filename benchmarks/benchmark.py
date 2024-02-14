import os
import time
import subprocess
import json
import sys
import glob
import shutil
import time
import pathlib

from models import CodeGen, CodeT5, StarCoder, DeepSeekCoder, CodeLlama, Bloom


class Benchmark:
    def __init__(self):
        pass


class HumanEval(Benchmark):
    def __init__(self, humaneval_dir, tmp_dir):
        super().__init__()
        self.humaneval_dir = humaneval_dir
        self.tmp_dir = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    def command_with_timeout(self, cmd, timeout=60):
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        t_beginning = time.time()
        while True:
            if p.poll() is not None:
                break
            seconds_passed = time.time() - t_beginning
            if timeout and seconds_passed > timeout:
                p.terminate()
                return 'TIMEOUT', 'TIMEOUT'
            time.sleep(1)
        out, err = p.communicate()
        return out, err

    def test_suite(self, algo, dir):
        CUR_DIR = os.getcwd()
        FNULL = open(os.devnull, 'w')
        try:
            os.chdir(dir)
            time.sleep(3)
            out, err = self.command_with_timeout(["mvn", "test", "-Dtest=TEST_" + algo.upper()], timeout=20)
            time.sleep(3)
            os.chdir(CUR_DIR)
            msg = (str(out) + str(err)).upper()
            if "compilation problems".upper() in msg or "compilation failure".upper() in msg:
                return 'uncompilable'
            elif "timeout".upper() in msg:
                return 'timeout'
            elif "build success".upper() in msg:
                return 'plausible'
            else:
                return "wrong"
        except Exception as e:
            print(e)
            os.chdir(CUR_DIR)
            return 'uncompilable'

    def insert_fix(self, filename, start_line, end_line, patch):
        """
        end_row is included in the buggy lines / buggy function
        """
        with open(filename, 'r') as file:
            data = file.readlines()

        with open(filename, 'w') as file:
            for i in range(start_line - 1):
                file.write(data[i] + '\n')
            file.write(patch.strip())
            for i in range(end_line, len(data)):
                file.write(data[i])

    def validate(self, input_file, output_file):
        # Copy directory
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        shutil.copytree(self.humaneval_dir, self.tmp_dir)

        plausible, total = 0, 0

        self.command_with_timeout(['rm', '-rf', self.tmp_dir + 'src/main/java/humaneval/buggy/'])
        self.command_with_timeout(['mkdir', self.tmp_dir + 'src/main/java/humaneval/buggy/'])
        self.command_with_timeout(['rm', '-rf', self.tmp_dir + 'src/test/java/humaneval/'])
        self.command_with_timeout(['mkdir', self.tmp_dir + 'src/test/java/humaneval/'])

        model_output = json.load(open(input_file, 'r'))
        validated_result = {'config': model_output['config'], 'data': {}}
        # validated_result = json.load(open(output_file, 'r'))
        for proj in model_output['data']:
            if proj in validated_result['data']:
                continue
            if 'output' not in model_output['data'][proj]:
                continue

            print('start validating', proj)
            total += 1

            self.command_with_timeout(['rm', '-rf', self.tmp_dir + 'src/main/java/humaneval/buggy/*.java'])
            self.command_with_timeout(['rm', '-rf', self.tmp_dir + 'src/test/java/humaneval/*.java'])
            shutil.copyfile(self.tmp_dir + 'src_bak/main/java/humaneval/buggy/' + proj + '.java',
                            self.tmp_dir + 'src/main/java/humaneval/buggy/' + proj + '.java')
            shutil.copyfile(self.tmp_dir + 'src_bak/test/java/humaneval/TEST_' + proj + '.java',
                            self.tmp_dir + 'src/test/java/humaneval/TEST_' + proj + '.java')

            validated_result['data'][proj] = {}
            for key, value in model_output['data'][proj].items():
                if key != 'output':
                    validated_result['data'][proj][key] = value
            validated_result['data'][proj]['output'] = []

            try:
                start_line, end_line = validated_result['data'][proj]['loc'].split('-')
                end_line = str(int(end_line) - 1) if end_line != start_line else end_line
                function_start_line, function_end_line = validated_result['data'][proj]['function range'].split('-')
                function_start_line, function_end_line = function_start_line.split(',')[0], \
                    function_end_line.split(',')[0]
            except Exception as e:
                continue

            current_is_correct = False
            for rank, patch in enumerate(model_output['data'][proj]['output']):
                filename = self.tmp_dir + 'src/main/java/humaneval/buggy/' + proj + '.java'
                if 'CODET5' in model_output['config']:
                    patch = CodeT5.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'CODEGEN' in model_output['config']:
                    patch = CodeGen.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(function_start_line), int(function_end_line), patch)
                elif 'STARCODER' in model_output['config']:
                    patch = StarCoder.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'DEEPSEEKCODER' in model_output['config']:
                    patch = DeepSeekCoder.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'BLOOM' in model_output['config']:
                    patch = Bloom.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(function_start_line), int(function_end_line), patch)
                elif 'CODELLAMA' in model_output['config']:
                    patch = CodeLlama.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                else:
                    assert False, 'unrecognized config.'

                correctness = self.test_suite(proj, self.tmp_dir)
                if correctness == 'plausible':
                    if not current_is_correct:
                        plausible += 1
                        current_is_correct = True
                    print(plausible, total, rank, "Plausible patch:", patch)
                elif correctness == 'wrong':
                    print(plausible, total, rank, "Wrong patch:", patch)
                elif correctness == 'timeout':
                    print(plausible, total, rank, "Timeout patch:", patch)
                elif correctness == 'uncompilable':
                    print(plausible, total, rank, "Uncompilable patch:", patch)
                validated_result['data'][proj]['output'].append({
                    'patch': patch, 'correctness': correctness
                })
                shutil.copyfile(self.tmp_dir + 'src_bak/main/java/humaneval/buggy/' + proj + '.java',
                                self.tmp_dir + 'src/main/java/humaneval/buggy/' + proj + '.java')
            json.dump(validated_result, open(output_file, 'w'), indent=2)

        # Remove temporary directory
        # shutil.rmtree(self.tmp_dir)


class QuixBugs(Benchmark):
    def __init__(self, quixbugs_dir, tmp_dir):
        super().__init__()
        self.quixbugs_dir = quixbugs_dir
        self.tmp_dir = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    def command_with_timeout(self, cmd, timeout=60):
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        t_beginning = time.time()
        while True:
            if p.poll() is not None:
                break
            seconds_passed = time.time() - t_beginning
            if timeout and seconds_passed > timeout:
                p.terminate()
                return 'TIMEOUT', 'TIMEOUT'
            time.sleep(1)
        out, err = p.communicate()
        return out, err

    def compile_fix(self, filename, tmp_dir):
        FNULL = open(os.devnull, 'w')
        p = subprocess.call(["javac",
                             tmp_dir + "Node.java",
                             tmp_dir + "WeightedEdge.java",
                             filename], stderr=FNULL)
        return False if p else True

    def test_suite(self, algo, dir):
        QUIXBUGS_MAIN_DIR = dir
        CUR_DIR = os.getcwd()
        FNULL = open(os.devnull, 'w')
        JAR_DIR = './'
        try:
            os.chdir(QUIXBUGS_MAIN_DIR)
            time.sleep(3)
            p1 = subprocess.Popen(["javac", "-cp", ".:java_programs:" + JAR_DIR + "junit4-4.12.jar:" + JAR_DIR +
                                   "hamcrest-all-1.3.jar", "java_testcases/junit/" + algo.upper() + "_TEST.java"],
                                  stdout=subprocess.PIPE, stderr=FNULL, universal_newlines=True)
            time.sleep(3)
            out, err = self.command_with_timeout(
                ["java", "-cp", ".:java_programs:" + JAR_DIR + "junit4-4.12.jar:" + JAR_DIR + "hamcrest-all-1.3.jar",
                 "org.junit.runner.JUnitCore", "java_testcases.junit." + algo.upper() + "_TEST"], timeout=10
            )
            print(out, err)
            os.chdir(CUR_DIR)
            if "FAILURES" in str(out) or "FAILURES" in str(err):
                return 'wrong'
            elif "TIMEOUT" in str(out) or "TIMEOUT" in str(err):
                return 'timeout'
            else:
                return 'plausible'
        except Exception as e:
            print(e)
            os.chdir(CUR_DIR)
            return 'uncompilable'

    def insert_fix(self, filename, start_line, end_line, patch):
        """
        end_row is included in the buggy lines / buggy function
        """
        with open(filename, 'r') as file:
            data = file.readlines()

        with open(filename, 'w') as file:
            for i in range(start_line - 1):
                file.write(data[i] + '\n')
            file.write(patch.strip())
            for i in range(end_line, len(data)):
                file.write(data[i])

    def validate(self, input_file, output_file):
        # Copy directory
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        shutil.copytree(self.quixbugs_dir, self.tmp_dir)

        plausible, total = 0, 0

        if not os.path.exists(self.tmp_dir):
            self.command_with_timeout(['mkdir', self.tmp_dir])

        model_output = json.load(open(input_file, 'r'))
        validated_result = {'config': model_output['config'], 'data': {}}
        for proj in model_output['data']:
            print('start validating', proj)
            total += 1
            self.command_with_timeout(['rm', '-rf', self.tmp_dir + '/java_programs/'])
            self.command_with_timeout(['mkdir', self.tmp_dir + '/java_programs/'])

            shutil.copyfile(self.tmp_dir + "/java_programs_bak/" + proj + '.java',
                            self.tmp_dir + "/java_programs/" + proj + '.java')
            shutil.copyfile(self.tmp_dir + "/java_programs_bak/Node.java", self.tmp_dir + "/java_programs/Node.java")
            shutil.copyfile(self.tmp_dir + "/java_programs_bak/WeightedEdge.java",
                            self.tmp_dir + "/java_programs/WeightedEdge.java")

            validated_result['data'][proj] = {}
            for key, value in model_output['data'][proj].items():
                if key != 'output':
                    validated_result['data'][proj][key] = value
            validated_result['data'][proj]['output'] = []
            try:
                start_line, end_line = validated_result['data'][proj]['loc'].split('-')
                end_line = str(int(end_line) - 1) if end_line != start_line else end_line
                function_start_line, function_end_line = validated_result['data'][proj]['function range'].split('-')
                function_start_line, function_end_line = function_start_line.split(',')[0], \
                    function_end_line.split(',')[0]
            except Exception as e:
                continue
            current_is_correct = False
            for rank, patch in enumerate(model_output['data'][proj]['output']):
                filename = self.tmp_dir + "/java_programs/" + proj + '.java'
                if 'CODET5' in model_output['config']:
                    patch = CodeT5.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'CODEGEN' in model_output['config']:
                    patch = CodeGen.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(function_start_line), int(function_end_line), patch)
                elif 'STARCODER' in model_output['config']:
                    patch = StarCoder.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'DEEPSEEKCODER' in model_output['config']:
                    patch = DeepSeekCoder.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'BLOOM' in model_output['config']:
                    patch = Bloom.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(function_start_line), int(function_end_line), patch)
                elif 'CODELLAMA' in model_output['config']:
                    patch = CodeLlama.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                else:
                    assert False, 'unrecognized config.'

                compile = self.compile_fix(filename, self.tmp_dir + "/java_programs/")
                correctness = 'uncompilable'
                if compile:
                    correctness = self.test_suite(proj, dir=self.tmp_dir)
                    if correctness == 'plausible':
                        if not current_is_correct:
                            plausible += 1
                            current_is_correct = True
                        print(plausible, total, rank, "Plausible patch:", patch)
                    elif correctness == 'wrong':
                        print(plausible, total, rank, "Wrong patch:", patch)
                    elif correctness == 'timeout':
                        print(plausible, total, rank, "Timeout patch:", patch)
                else:
                    print(plausible, total, rank, 'Uncompilable patch:', patch)
                validated_result['data'][proj]['output'].append({
                    'patch': patch, 'correctness': correctness
                })
                shutil.copyfile(self.tmp_dir + "/java_programs_bak/" + proj + '.java',
                                self.tmp_dir + "/java_programs/" + proj + '.java')
            json.dump(validated_result, open(output_file, 'w'), indent=2)

        # Remove directory
        shutil.rmtree(self.tmp_dir)


class Defects4j(Benchmark):
    def __init__(self, defects4j_dir, tmp_dir):
        super().__init__()
        self.defects4j_dir = defects4j_dir
        self.tmp_counter = 0
        self.tmp_root = tmp_dir[:-1]
        print(f'Root: {self.tmp_root}')
        self.tmp_dir = f"{self.tmp_root}/{self.tmp_counter}/"
        # Create dir
        pathlib.Path(self.tmp_root).mkdir(parents=True, exist_ok=True)
        # pathlib.Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

    def create_tmp_folder(self, tmp_dir):
        tmp_dir = f"{self.tmp_root}/{self.tmp_counter}/"
        os.makedirs(tmp_dir)
        self.tmp_counter += 1
        return tmp_dir

    def checkout_project(self, project, bug_id, tmp_dir):
        FNULL = open(os.devnull, 'w')
        command = "defects4j checkout " + " -p " + project + " -v " + bug_id + " -w " + tmp_dir
        p = subprocess.Popen([command], shell=True, stdout=FNULL, stderr=FNULL)
        p.wait()

    def compile_fix(self, project_dir):
        os.chdir(project_dir)
        p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if "FAIL" in str(err) or "FAIL" in str(out):
            return False
        return True

    def command_with_timeout(self, cmd, timeout=300):
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        t_beginning = time.time()
        while True:
            if p.poll() is not None:
                break
            seconds_passed = time.time() - t_beginning
            if timeout and seconds_passed > timeout:
                p.terminate()
                return 'TIMEOUT', 'TIMEOUT'
            time.sleep(1)
        out, err = p.communicate()
        return out, err

    def test_suite(self, project_dir, timeout=300):
        os.chdir(project_dir)
        out, err = self.command_with_timeout(["defects4j", "test", "-r"], timeout)
        return out, err

    def trigger(self, project_dir, timeout=300):
        os.chdir(project_dir)
        out, err = self.command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout)
        return out, err

    def relevant(self, project_dir, timeout=300):
        os.chdir(project_dir)
        out, err = self.command_with_timeout(["defects4j", "export", "-p", "tests.relevant"], timeout)
        return out, err

    def test_one(self, project_dir, test_case, timeout=300):
        os.chdir(project_dir)
        out, err = self.command_with_timeout(["defects4j", "test", "-t", test_case], timeout)
        return out, err

    def insert_fix(self, filename, start_line, end_line, patch):
        """
        end_row is included in the buggy lines / buggy function
        """
        with open(filename, 'r') as file:
            data = file.readlines()

        with open(filename, 'w') as file:
            for i in range(start_line - 1):
                file.write(data[i])
            file.write(patch.strip() + '\n')
            for i in range(end_line, len(data)):
                file.write(data[i])

    def validate(self, input_file, output_file):
        plausible, total = 0, 0
        model_output = json.load(open(input_file, 'r'))
        validated_result = {'config': model_output['config'], 'data': {}}
        # validated_result = json.load(open(output_file, 'r'))
        for key in model_output['data']:
            if key in validated_result['data']:
                continue
            if 'output' not in model_output['data'][key]:
                continue

            key_list = key.split('_')
            proj, bug_id, loc = key_list[0], key_list[1], key_list[-1]
            path = '_'.join(key_list[2: -1])

            print('start validating', proj, bug_id)
            total += 1

            validated_result['data'][key] = {}
            for k, value in model_output['data'][key].items():
                if k != 'output':
                    validated_result['data'][key][k] = value
            validated_result['data'][key]['output'] = []

            try:
                start_line, end_line = validated_result['data'][key]['loc'].split('-')
                end_line = str(int(end_line) - 1) if end_line != start_line else end_line
                function_start_line, function_end_line = validated_result['data'][key]['function range'].split('-')
                function_start_line, function_end_line = function_start_line.split(',')[0], \
                    function_end_line.split(',')[0]
            except:
                print("Couldn't be compiled")
                continue

            self.tmp_dir = self.create_tmp_folder(self.tmp_dir)
            print(self.tmp_dir)
            self.checkout_project(proj, bug_id + 'b', self.tmp_dir)
            if proj == "Mockito":
                print("Mockito needs separate compilation")
                self.compile_fix(self.tmp_dir)

            # check standard test time
            start_time = time.time()
            init_out, init_err = self.test_suite(self.tmp_dir)
            standard_time = int(time.time() - start_time)

            # check failed test cases
            failed_test_cases = str(init_out).split(' - ')[1:]
            for i, failed_test_case in enumerate(failed_test_cases):
                failed_test_cases[i] = failed_test_case.strip()
            init_fail_num = len(failed_test_cases)
            print(init_fail_num, str(standard_time) + 's')

            # check triggering failed test cases
            trigger, err = self.trigger(self.tmp_dir)
            triggers = trigger.strip().split('\n')
            for i, trigger in enumerate(triggers):
                triggers[i] = trigger.strip()
            print('trigger number:', len(triggers))

            current_is_correct = False
            for rank, patch in enumerate(model_output['data'][key]['output']):
                filename = self.tmp_dir + path
                shutil.copyfile(filename, filename + '.bak')

                if 'CODET5' in model_output['config']:
                    patch = CodeT5.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'CODEGEN' in model_output['config']:
                    patch = CodeGen.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(function_start_line), int(function_end_line), patch)
                elif 'STARCODER' in model_output['config']:
                    patch = StarCoder.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'DEEPSEEKCODER' in model_output['config']:
                    patch = DeepSeekCoder.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                elif 'BLOOM' in model_output['config']:
                    patch = Bloom.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(function_start_line), int(function_end_line), patch)
                elif 'CODELLAMA' in model_output['config']:
                    patch = CodeLlama.output_to_patch(patch, model_output['config'])
                    self.insert_fix(filename, int(start_line), int(end_line), patch)
                else:
                    assert False, 'unrecognized config.'

                if proj == 'Mockito':
                    # Mockito needs seperate compile
                    self.compile_fix(self.tmp_dir)

                # trigger cases is few and total time is long, we test trigger cases first.
                outs = []
                correctness = None
                start_time = time.time()
                if standard_time >= 10 and len(triggers) <= 5:
                    for trigger in triggers:
                        out, err = self.test_one(self.tmp_dir, trigger,
                                                timeout=min(200, int(1.5 * standard_time)))
                        if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                            print(plausible, total, rank, 'Time out for patch: ', patch,
                                str(int(time.time() - start_time)) + 's')
                            correctness = 'timeout'
                            break
                        elif 'FAIL' in str(err) or 'FAIL' in str(out):
                            print(plausible, total, rank, 'Uncompilable patch:', patch,
                                str(int(time.time() - start_time)) + 's')
                            correctness = 'uncompilable'
                            break
                        elif "Failing tests: 0" in str(out):
                            continue
                        else:
                            outs += str(out).split(' - ')[1:]
                if len(set(outs)) >= len(triggers):
                    # does not pass any one more
                    print(plausible, total, rank, 'Wrong patch:', patch,
                        str(int(time.time() - start_time)) + 's')
                    correctness = 'wrong'

                if correctness is None:
                    # pass at least one more trigger case
                    # have to pass all non-trigger
                    out, err = self.test_suite(self.tmp_dir, timeout=min(200, int(1.5 * standard_time)))

                    if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                        print(plausible, total, rank, 'Time out for patch: ', patch,
                            str(int(time.time() - start_time)) + 's')
                        correctness = 'timeout'
                    elif 'FAIL' in str(err) or 'FAIL' in str(out):
                        print(plausible, total, rank, 'Uncompilable patch:', patch,
                            str(int(time.time() - start_time)) + 's')
                        correctness = 'uncompilable'
                    elif "Failing tests: 0" in str(out):
                        if not current_is_correct:
                            current_is_correct = True
                            plausible += 1
                        print(plausible, total, rank, 'Plausible patch:', patch,
                            str(int(time.time() - start_time)) + 's')
                        correctness = 'plausible'
                    elif len(str(out).split(' - ')[1:]) < init_fail_num:
                        # fail less, could be correct
                        current_failed_test_cases = str(out).split(' - ')[1:]
                        no_new_fail = True
                        for current_failed_test_case in current_failed_test_cases:
                            if current_failed_test_case.strip() not in failed_test_cases:
                                no_new_fail = False
                                break
                        if no_new_fail:
                            # fail less and no new fail cases, could be plausible
                            if not current_is_correct:
                                current_is_correct = True
                                plausible += 1
                            print(plausible, total, rank, 'Plausible patch:', patch,
                                str(int(time.time() - start_time)) + 's')
                            correctness = 'plausible'
                        else:
                            print(plausible, total, rank, 'Wrong patch:', patch,
                                str(int(time.time() - start_time)) + 's')
                            correctness = 'wrong'
                    else:
                        print(plausible, total, rank, 'Wrong patch:', patch,
                            str(int(time.time() - start_time)) + 's')
                        correctness = 'wrong'

                validated_result['data'][key]['output'].append({
                    'patch': patch, 'correctness': correctness
                })
                shutil.copyfile(filename + '.bak', filename)

            # write after finish validating every bug, to avoid wasting time
            json.dump(validated_result, open(output_file, 'w'), indent=2)

        # write the last time after validating all
        json.dump(validated_result, open(output_file, 'w'), indent=2)

        # Remove temporary directory
        # shutil.rmtree(self.tmp_root, ignore_errors=True)
