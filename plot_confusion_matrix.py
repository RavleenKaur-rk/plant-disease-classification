
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load confusion matrix
cm = np.loadtxt("confusion_matrix.csv", delimiter=",", dtype=int)

# Load class names in the same order as the model
with open("labels_15.txt", "r") as file:
    labels = [line.strip() for line in file if line.strip()]

# Verify dimensions
assert cm.shape == (15, 15), f"Unexpected matrix shape: {cm.shape}"
assert len(labels) == 15, f"Expected 15 labels, found {len(labels)}"

# Create heatmap
fig, ax = plt.subplots(figsize=(18, 16))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    cmap="Blues",
    values_format="d",
    xticks_rotation=90,
    ax=ax,
    colorbar=True
)

plt.title("Plant Disease Classification - Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

# Save graph
plt.savefig("confusion_matrix_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("Confusion matrix heatmap saved successfully!")
