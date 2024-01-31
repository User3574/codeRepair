import os
import sys
import json
import codecs
import subprocess
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM


# Model Name, Benchmark name
if __name__ == '__main__':
    ckpt = "Salesforce/codet5p-220m"
    input_file = "/home/machacini/codeRepair/benchmarks/results/quixbugs/deepseekcoder/input_c1.json"

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to("cuda")
    bloom_output = json.load(open(input_file, 'r'))
    for i, filename in enumerate(bloom_output['data']):
        text = bloom_output['data'][filename]['input']
        print(i + 1, 'generating', filename)

        try:
            text = "public static boolean has_close_elements(List<Double> numbers, double threshold){\n    for (int i = 0; i < numbers.size(); i += 1){\n        for (int j = i + 1; j < numbers.size(); j += 1){\n            <extra_id_0>\n            if (distance < threshold)\n                return true;\n        }\n    }\n    return false;\n}"
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            inputs['decoder_input_ids'] = inputs['input_ids'].clone()

            # Original was 512
            if inputs['input_ids'].size(1) >= 100:
                print('input too long:', inputs['input_ids'].size(1), 'skip')
                continue

            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            generated_ids = model.generate(**inputs, max_new_tokens=4, num_beams=10, num_return_sequences=10,eos_token_id=eos_id)
            output = []
            for generated_id in generated_ids:
                o = tokenizer.decode(generated_id, skip_special_tokens=True)
                print(o)
                output.append(o)
            break
        except Exception as e:
            print(f"Can't load the model, unexpected exception occured: {e}")
            output = []
