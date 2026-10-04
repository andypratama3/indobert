import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATA_PATH = Path(
    "data/processed/dataset_train_clean.csv"
)

MODEL_NAME = "indobenchmark/indobert-base-p1"

# Folder utama untuk semua hasil cross-validation
CV_OUTPUT_DIR = Path(
    "models/indobert_aspect_sentiment_cv"
)

CV_RESULT_DIR = Path(
    "data/results/cv"
)

N_SPLITS = 10
N_EPOCHS = 7
RANDOM_STATE = 42

LABEL_MAP = {
    "transparansi_positif": 0,
    "transparansi_negatif": 1,
    "akuntabilitas_positif": 2,
    "akuntabilitas_negatif": 3,
    "efektivitas_efisiensi_positif": 4,
    "efektivitas_efisiensi_negatif": 5,
    "responsivitas_positif": 6,
    "responsivitas_negatif": 7,
}

ID_TO_LABEL = {
    v: k for k, v in LABEL_MAP.items()
}


class AspectSentimentDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=256,
    ):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value
            in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item

    def __len__(self):

        return len(self.labels)


def compute_metrics(pred):

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        )
    )

    acc = accuracy_score(
        labels,
        preds,
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_and_clean_data():
    """
    Sama persis dengan logika di train_indobert.py:
    load csv, validasi kolom, normalisasi label, mapping ke id.
    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"File tidak ditemukan:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "clean_text",
        "final_label",
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Kolom '{col}' tidak ditemukan."
            )

    df = df.dropna(
        subset=[
            "clean_text",
            "final_aspect",
            "final_sentiment",
        ]
    ).copy()

    df["clean_text"] = (
        df["clean_text"]
        .astype(str)
    )

    df["final_aspect"] = (
        df["final_aspect"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["final_sentiment"] = (
        df["final_sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["final_label"] = (
        df["final_label"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" & ", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    df = df[
        df["final_label"]
        .isin(LABEL_MAP.keys())
    ].copy()

    if df.empty:

        raise ValueError(
            "Tidak ada data valid untuk training."
        )

    df["label"] = (
        df["final_label"]
        .map(LABEL_MAP)
    )

    label_counts = (
        df["final_label"]
        .value_counts()
    )

    print("\nDistribusi Label (seluruh dataset):")
    print(label_counts)

    if len(df) < 10:

        raise ValueError(
            "Jumlah data terlalu sedikit."
        )

    if label_counts.min() < N_SPLITS:

        print(
            "\n[PERINGATAN] Ada kelas dengan jumlah data "
            f"< {N_SPLITS} (jumlah fold). StratifiedKFold tetap bisa "
            "berjalan, tapi pastikan tidak ada kelas dengan data "
            "kurang dari jumlah fold."
        )

    return df


def run_fold(fold_idx, train_texts, train_labels, val_texts, val_labels, tokenizer):

    fold_output_dir = CV_OUTPUT_DIR / f"fold_{fold_idx}"
    fold_logging_dir = Path("logs/indobert_aspect_sentiment_cv") / f"fold_{fold_idx}"

    train_dataset = AspectSentimentDataset(
        train_texts, train_labels, tokenizer
    )

    val_dataset = AspectSentimentDataset(
        val_texts, val_labels, tokenizer
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_MAP),
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=N_EPOCHS,
        weight_decay=0.01,
        logging_dir=str(fold_logging_dir),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Fold {fold_idx}/{N_SPLITS}: mulai training ===")

    trainer.train()

    eval_result = trainer.evaluate()
    eval_result["fold"] = fold_idx

    predictions = trainer.predict(val_dataset)

    y_true = predictions.label_ids
    y_pred = predictions.predictions.argmax(axis=1)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_MAP))),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))],
        columns=[ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))],
    )

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    CV_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(fold_output_dir))
    tokenizer.save_pretrained(str(fold_output_dir))

    cm_path = CV_RESULT_DIR / f"confusion_matrix_fold_{fold_idx}.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print(f"=== Fold {fold_idx} selesai. Model: {fold_output_dir} ===")
    print(eval_result)

    # Bersihkan memori GPU/CPU sebelum lanjut ke fold berikutnya
    del trainer
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return eval_result


def main():

    df = load_and_clean_data()

    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels)

    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(texts_arr, labels_arr), start=1
    ):

        train_texts = texts_arr[train_idx].tolist()
        train_labels = labels_arr[train_idx].tolist()

        val_texts = texts_arr[val_idx].tolist()
        val_labels = labels_arr[val_idx].tolist()

        fold_metrics = run_fold(
            fold_idx,
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            tokenizer,
        )

        all_fold_metrics.append(fold_metrics)

    # Simpan hasil semua fold
    cv_df = pd.DataFrame(all_fold_metrics)

    CV_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    per_fold_path = CV_RESULT_DIR / "cv_per_fold_metrics.csv"
    cv_df.to_csv(per_fold_path, index=False, encoding="utf-8-sig")

    # Hitung rata-rata & std dari metrik utama
    metric_cols = [
        c for c in cv_df.columns
        if c.startswith("eval_") and c not in ("eval_runtime",)
    ]

    summary = cv_df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    summary_path = CV_RESULT_DIR / "cv_summary_metrics.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print("\n\n================ RINGKASAN 10-FOLD CROSS VALIDATION ================")
    print(summary)
    print(f"\nDetail per fold disimpan di: {per_fold_path}")
    print(f"Ringkasan rata-rata/std disimpan di: {summary_path}")
    print(f"Model tiap fold disimpan di folder: {CV_OUTPUT_DIR}/fold_<n>")


if __name__ == "__main__":
    main()