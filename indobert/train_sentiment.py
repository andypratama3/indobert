import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
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
    "data/processed/aspect_annotation_final_clean.csv"
)

MODEL_NAME = "indobenchmark/indobert-base-p1"

OUTPUT_DIR = Path(
    "models/indobert_sentiment"
)

RESULT_PATH = Path(
    "data/results/indobert_sentiment_evaluation.csv"
)

LABEL_MAP = {
    "positif": 0,
    "negatif": 1,
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


def main():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"File tidak ditemukan:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "clean_text",
        "final_aspect",
        "final_sentiment",
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
        df["final_sentiment"]
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

    print("\nDistribusi Label:")
    print(label_counts)

    if len(df) < 10:

        raise ValueError(
            "Jumlah data terlalu sedikit."
        )

    if label_counts.min() < 2:

        raise ValueError(
            "Ada kelas yang hanya memiliki 1 data."
        )

    texts = (
        df["clean_text"]
        .tolist()
    )

    labels = (
        df["label"]
        .tolist()
    )

    (
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = (
        AspectSentimentDataset(
            train_texts,
            train_labels,
            tokenizer,
        )
    )

    test_dataset = (
        AspectSentimentDataset(
            test_texts,
            test_labels,
            tokenizer,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(
                LABEL_MAP
            ),
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
        )
    )

    training_args = (
        TrainingArguments(
            output_dir=str(
                OUTPUT_DIR
            ),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=7,
            weight_decay=0.01,
            logging_dir=
            "logs/indobert_sentiment",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(
        "\nMulai Fine-Tuning IndoBERT..."
    )

    trainer.train()

    eval_result = (
        trainer.evaluate()
    )

    predictions = trainer.predict(
    test_dataset
    )
    
    y_true = predictions.label_ids
    
    y_pred = (
    predictions
    .predictions
    .argmax(axis=1)
    )
    
    cm = confusion_matrix(
    y_true,
    y_pred,
    )
    
    cm_df = pd.DataFrame(
    cm,
    index=[
        ID_TO_LABEL[i]
        for i in range(
            len(ID_TO_LABEL)
        )
    ],
    columns=[
        ID_TO_LABEL[i]
        for i in range(
            len(ID_TO_LABEL)
        )
    ],
    )
    CM_PATH = Path(
    "data/results/confusion_matrix_sentiment.csv"
    )

    cm_df.to_csv(
    CM_PATH,
    encoding="utf-8-sig",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    pd.DataFrame(
        [eval_result]
    ).to_csv(
        RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nFine-tuning selesai."
    )

    print(
        f"\nModel disimpan di:\n{OUTPUT_DIR}"
    )

    print(
        f"\nHasil evaluasi:\n{RESULT_PATH}"
    )

    print(
    f"\nConfusion Matrix:\n{CM_PATH}"
    )

    print(
        "\nMetric:"
    )

    print(
        eval_result
    )


if __name__ == "__main__":
    main()