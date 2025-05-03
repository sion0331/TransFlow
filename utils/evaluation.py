from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate(model, loader, device, title="Confusion Matrix"):
    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.int64)
            outputs = model(X_batch)
            _, y_pred = outputs.max(1)

            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()

    # Metrics
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, average='macro')
    recall = recall_score(all_y_true, all_y_pred, average='macro')
    f1 = f1_score(all_y_true, all_y_pred, average='macro')

    print(classification_report(all_y_true, all_y_pred, digits=4))
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision (macro): {precision:.4f}")
    print(f"Test Recall (macro): {recall:.4f}")
    print(f"Test F1 Score (macro): {f1:.4f}")

    # return {
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall": recall,
    #     "f1": f1
    # }
