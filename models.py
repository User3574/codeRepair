from transformers import AutoModel, T5ForConditionalGeneration, RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, TrainingArguments, Trainer


def get_model(model_name, checkpoint):
    # Load pretrained model, tokenizer
    if model_name == "CodeT5":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', return_dict=True)
    elif model_name == "CodeLlama":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = LlamaForCausalLM.from_pretrained(checkpoint, device_map='auto')
    elif model_name == "StarCoder":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')
    elif model_name == "CodeGEN":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')
    elif model_name == "InCoder":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')
    elif model_name == "CodeBERT":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto', return_dict=True)
        
    return model, tokenizer
