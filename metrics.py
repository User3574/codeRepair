import numpy as np
import evaluate

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def prepare_compute_metrics(tokenizer, hf_metrics):
    def compute_metrics(eval_preds):
        nonlocal tokenizer, hf_metrics

        preds, labels = eval_preds

        # Remove Collate
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Print for Debug
        print(f'Preds: {preds[0]}, Decoded: {decoded_preds[0]}')
        print(f'Labels: {labels[0]}, Decoded: {decoded_labels[0]}')

        # Compute Metrics
        results = {}
        for metric in hf_metrics:
            print(f"Computing metric: {hf_metrics[metric]['name']}")
            loaded_metric = evaluate.load(hf_metrics[metric]["name"])
            results[metric] = loaded_metric.compute(predictions=decoded_preds, references=decoded_labels, **hf_metrics[metric]["kwargs"])

        # Calculate Sentence Accuracy
        print('Predictions:', decoded_preds[0])
        print('Labels:', decoded_labels[0])
        return results

    return compute_metrics
