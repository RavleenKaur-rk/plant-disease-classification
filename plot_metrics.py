
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Load confusion matrix
cm = np.loadtxt("confusion_matrix.csv", delimiter=",", dtype=int)

# Load class labels
with open("labels_15.txt", "r") as file:
    labels = [line.strip() for line in file if line.strip()]

# Overall evaluation metrics
metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
metric_values = [91.78, 92.24, 91.78, 91.77]

plt.figure(figsize=(9, 5))
bars = plt.bar(metric_names, metric_values)

plt.ylim(0, 100)
plt.ylabel("Score (%)")
plt.title("Overall Model Performance")
plt.bar_label(bars, fmt="%.2f%%", padding=3)
plt.tight_layout()
plt.savefig("overall_metrics.png", dpi=300)
plt.show()

# Class-wise precision, recall and F1-score
precision, recall, f1, support = precision_recall_fscore_support(
    np.repeat(np.arange(len(labels)), cm.sum(axis=1)),
    np.repeat(np.arange(len(labels)), cm.sum(axis=1)),
    zero_division=0
) if False else (None, None, None, None)

# Calculate metrics directly from confusion matrix
true_positive = np.diag(cm)
actual_count = cm.sum(axis=1)
predicted_count = cm.sum(axis=0)

precision = np.divide(
    true_positive, predicted_count,
    out=np.zeros(len(labels), dtype=float),
    where=predicted_count != 0
)

recall = np.divide(
    true_positive, actual_count,
    out=np.zeros(len(labels), dtype=float),
    where=actual_count != 0
)

f1 = np.divide(
    2 * precision * recall, precision + recall,
    out=np.zeros(len(labels), dtype=float),
    where=(precision + recall) != 0
)

plt.figure(figsize=(14, 7))
bars = plt.bar(range(len(labels)), f1 * 100)

plt.xticks(range(len(labels)), labels, rotation=90)
plt.ylabel("F1-score (%)")
plt.title("Class-wise F1-score")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("classwise_f1.png", dpi=300)
plt.show()

print("Both performance graphs saved successfully!")
