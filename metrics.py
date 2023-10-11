import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def compute_metrics(batch):
    pred, labels = batch
    pred = np.argmax(pred, axis=1)

    # Basic metrics
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    # TODO: CodeBLEU, BLEU
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}