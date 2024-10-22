# Code for Fine-tuning Language models for Program Repair: An Empirical Study
This repo contains code for the work done at Simula and UiO focused on fine-tuning LLMs using standard full fine-tuning approach along with parameter-efficient fine-tuning (PEFT) techniques which are then compared on several benchmarks. Work is based on the paper and repo by Jiang et al. [paper](https://arxiv.org/pdf/2302.05020), [codebase](https://github.com/lin-tan/clm/tree/main).

## Requirements
There are various requirements in order to run this project. 
Most importantly, [Quixbugs](https://github.com/jkoppel/QuixBugs), 
[Defects4j](https://github.com/rjust/defects4j) and [HumanEval-Java](https://zenodo.org/records/7559208)
benchmarks have to be downloaded and installed. Furthermore, jesper is needed to be extracted from the repo in order to compile the code in Java. 

Afterwards we can install requirements ```requirements.txt``` with pip and run the selected code.
There are several files to be ran including for example
- Fine-tuning of various models using ```finetune.py```
- Benchmarking of various models using ```benchmarks/validate.py```
- Prepare inputs and outputs of models using ```benchmarks/generate_inputs.py```, ```benchmarks/generate_outputs.py```
- Perform post-processing of results by ```benchmarks/analyse.py```
- PEFT fine-tuning with ```lora.py``` and ```ia3.py```
