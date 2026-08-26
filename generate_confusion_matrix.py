import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# ==================================================
# OFFICIAL CONFUSION MATRIX
# ==================================================

cm = np.array([
    [159, 56, 43, 55, 34, 14, 4],
    [49, 183, 26, 43, 22, 6, 16],
    [33, 48, 93, 58, 68, 33, 12],
    [82, 69, 36, 108, 19, 18, 8],
    [33, 17, 48, 25, 169, 40, 13],
    [12, 14, 34, 26, 46, 215, 18],
    [6, 34, 3, 23, 4, 8, 317]
])


class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ==================================================
# CREATE FIGURE
# ==================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "HAM10000 5-Way 5-Shot Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)

plt.tight_layout()


# ==================================================
# SAVE
# ==================================================

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Confusion matrix saved successfully:"
)

print(
    "results/confusion_matrix.png"
)