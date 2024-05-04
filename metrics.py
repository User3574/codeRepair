import numpy as np
import evaluate

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def prepare_compute_metrics(tokenizer, hf_metrics, is_shifted):
    def compute_metrics(eval_preds):
        nonlocal tokenizer, hf_metrics, is_shifted

        preds, targets = eval_preds

        # Shift labels to the left
        if is_shifted == True:
            targets = np.roll(targets, -1)
        
        # Remove Collate
        labels = np.where(targets != -100, targets, tokenizer.pad_token_id)
        preds = np.where(targets != -100, preds, tokenizer.pad_token_id)

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute Metrics
        results = {}
        for metric in hf_metrics:
            print(f"Computing metric: {hf_metrics[metric]['name']}")
            loaded_metric = evaluate.load(hf_metrics[metric]["name"])
            results[metric] = loaded_metric.compute(predictions=decoded_preds, references=decoded_labels, **hf_metrics[metric]["kwargs"])

        # Print sample
        print('Predictions:\n', decoded_preds[0])
        print('Labels:\n', decoded_labels[0])
        return results

    return compute_metrics
