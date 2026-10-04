import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# Folder Input & Output
# ==============================

INPUT_DIR = Path("data/results/cv")

OUTPUT_DIR = Path(
    "data/results/visualization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==============================
# Generate Semua Fold
# ==============================

for fold in range(1, 11):

    cm_path = (
        INPUT_DIR /
        f"confusion_matrix_fold_{fold}.csv"
    )

    if not cm_path.exists():

        print(
            f"Fold {fold} tidak ditemukan."
        )

        continue

    df = pd.read_csv(
        cm_path,
        index_col=0
    )

    plt.figure(
        figsize=(10,8)
    )

    plt.imshow(
        df,
        cmap="Blues"
    )

    plt.colorbar()

    plt.xticks(
        range(len(df.columns)),
        df.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(df.index)),
        df.index
    )

    for i in range(len(df.index)):
        for j in range(len(df.columns)):

            plt.text(
                j,
                i,
                df.iloc[i, j],
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.title(
        f"Confusion Matrix Fold {fold}"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"confusion_matrix_fold_{fold}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Fold {fold} selesai"
    )

print(
    "\nSemua confusion matrix berhasil dibuat."
)

print(
    f"Hasil berada di:\n{OUTPUT_DIR}"
)