import numpy as np
import evaluate

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def prepare_compute_metrics(tokenizer, hf_metrics):
    def compute_metrics(eval_preds):
        nonlocal tokenizer, hf_metrics
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        # preds have the same shape as the labels due to preprocess_logits_for_metrics

        # Masking
        # mask = labels != -100
        # labels = labels[mask]
        # preds = preds[mask]

        results = {}
        preds = tokenizer.batch_decode(preds)
        labels = tokenizer.batch_decode(labels)

        # Work with HuggingFace Metrics
        for metric in hf_metrics:
            print(f"Computing metric: {hf_metrics[metric]['name']}")
            loaded_metric = evaluate.load(hf_metrics[metric]["name"])
            results[metric] = loaded_metric.compute(predictions=preds, references=labels, **hf_metrics[metric]["kwargs"])

        # Calculate Accuracy
        equal_sentences = []
        print(labels[0], preds[0])
        for i in range(len(labels)):
            equal_sentences.append(labels[i] == preds[i])
        results["accuracy"] = sum(equal_sentences)/len(equal_sentences)

        return results

    return compute_metrics
