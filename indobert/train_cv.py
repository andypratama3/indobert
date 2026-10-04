import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

# Berapa banyak contoh validasi minimum yang boleh diharapkan ada
# di setiap fold. StratifiedKFold hanya bisa menyebar ke semua
# fold jika jumlah anggota kelas >= N_SPLITS.
MIN_PER_FOLD = N_SPLITS

# Diisi oleh load_and_clean_data(): daftar label yang jumlahnya
# < N_SPLITS, sehingga TIDAK bisa muncul di setiap fold.
# Diekspos ke script lain (mis. analysis/class_imbalance_report.py).
SPARSE_CLASSES = []

# Signature komposisi kelas validasi yang sudah dicetak. Satu fold
# dievaluasi N_EPOCHS kali; laporan hanya perlu dicetak sekali
# per komposisi fold yang berbeda.
_REPORTED_FOLD_SIGNATURES = set()


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


def get_sparse_classes(label_counts, n_splits=N_SPLITS):
    """
    Kelas yang jumlah anggotanya < n_splits.

    StratifiedKFold hanya bisa menugaskan minimal satu anggota
    kelas ke tiap fold. Kalau jumlah anggota kelas lebih kecil
    dari n_splits, beberapa fold PASTI kosong untuk kelas itu,
    jadi precision/recall/F1 kelas tersebut tidak bisa dihitung
    di fold-fold itu.

    Return: list of (label, count), diurutkan dari yang paling langka.
    """

    return [
        (name, int(count))
        for name, count in label_counts.items()
        if int(count) < n_splits
    ]


def print_sparse_class_warning(
    sparse_classes,
    n_rows,
    n_splits=N_SPLITS,
):

    """
    Peringatan eksplisit + bisa ditindaklanjuti, bukan sekadar
    "pastikan tidak ada kelas dengan data kurang dari jumlah fold".
    Pipeline tetap dilanjutkan; tugasnya memberi tahu dampaknya.
    """

    print(
        "\n" + "!" * 68
    )

    print(
        f"[PERINGATAN PENTING] {len(sparse_classes)} kelas memiliki "
        f"< {n_splits} data ({n_rows} baris total), sehingga "
        "TIDAK bisa di-stratify ke semua fold."
    )

    print(
        f"{'label':<34}{'jumlah':>8}{'butuh min':>11}"
    )

    for name, count in sorted(
        sparse_classes,
        key=lambda item: item[1],
    ):

        print(
            f"{name:<34}{count:>8}{n_splits:>11}"
        )

    print(
        "  Dampak yang terjadi:\n"
        f"  - SETIAP kelas di atas akan bernilai 0 pada minimal "
        f"1 dari {n_splits} fold validasi.\n"
        "  - Pada fold itu, precision/recall/F1 kelas tersebut "
        "0.000 (support 0).\n"
        "  - Rata-rata berbobot (weighted) tetap dihitung dan "
        "tidak error,\n"
        "    tapi ia menutupi kelas yang benar-benar tidak "
        "terlihat sama sekali.\n"
        "  - Std antar fold membesar karena fold dengan dan "
        "tanpa kelas langka\n"
        "    tidak bisa dibandingkan secara apples-to-apples."
    )

    print(
        "\n  Tindakan yang disarankan (lihat docs/BUG_TRACKER.md BUG-08):\n"
        f"  1. Turunkan n_splits sehingga n_splits <= jumlah "
        f"kelas terkecil\n"
        f"     (mis. n_splits={min(c for _, c in sparse_classes)} "
        "pada data saat ini).\n"
        "  2. Gabungkan kelas positif yang sangat langka "
        "(mis. *_positif\n"
        "     untuk Responsivitas dan Transparansi).\n"
        "  3. Atau pertahankan n_splits=10 dan LAPORKAN metrik "
        "per kelas dengan\n"
        "     caveat eksplisit. Jalankan:\n"
        "     python -m analysis.class_imbalance_report"
    )

    print(
        "!" * 68
    )


def build_fold_coverage_table(
    labels_arr,
    skf=None,
    n_splits=N_SPLITS,
):

    """
    Hitung jumlah contoh validasi tiap kelas pada tiap fold.

    Return: DataFrame dengan kolom
        fold, label_id, label, n_train, n_val
    """

    if skf is None:

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    texts_arr = np.zeros(
        len(labels_arr),
        dtype=object,
    )

    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(texts_arr, labels_arr),
        start=1,
    ):

        val_counts = (
            np.bincount(
                np.asarray(labels_arr)[val_idx],
                minlength=len(LABEL_MAP),
            )
        )

        train_counts = (
            np.bincount(
                np.asarray(labels_arr)[train_idx],
                minlength=len(LABEL_MAP),
            )
        )

        for label_id in range(len(LABEL_MAP)):

            rows.append(
                {
                    "fold": fold_idx,
                    "label_id": label_id,
                    "label": ID_TO_LABEL[label_id],
                    "n_train": int(train_counts[label_id]),
                    "n_val": int(val_counts[label_id]),
                }
            )

    return pd.DataFrame(rows)


def print_fold_integrity_warning(coverage_df, n_splits=N_SPLITS):

    """
    Cetak tabel peringatan untuk setiap pasangan (fold, kelas)
    yang tidak punya satupun contoh validasi.
    """

    missing = coverage_df[
        coverage_df["n_val"] == 0
    ].sort_values(
        ["fold", "label_id"]
    )

    print(
        "\n" + "=" * 68
    )

    print(
        "CEK INTEGRITAS FOLD  "
        "(pasangan fold x kelas dengan 0 contoh validasi)"
    )

    print("=" * 68)

    if missing.empty:

        print(
            "  Tidak ada. Semua kelas muncul di setiap fold."
        )

    else:

        print(
            f"  {len(missing)} pasangan (fold, kelas) "
            f"tidak punya contoh validasi:\n"
        )

        print(
            f"{'fold':>6}{'label_id':>11}"
            f"{'label':>34}{'n_val':>7}"
        )

        for _, row in missing.iterrows():

            print(
                f"{int(row['fold']):>6}"
                f"{int(row['label_id']):>11}"
                f"{row['label']:>34}"
                f"{int(row['n_val']):>7}"
            )

        affected = sorted(
            missing["label"].unique()
        )

        print(
            f"\n  Kelas terdampak: {len(affected)} dari "
            f"{len(LABEL_MAP)} -> {affected}"
        )

        print(
            f"\n  Pada {missing['fold'].nunique()} dari {n_splits} "
            "fold, ada kelas yang absen total dari validasi.\n"
            "  Metrik kelas tersebut di fold itu tidak "
            "mengukuran apa pun -- bukan karena\n"
            "  model buruk, tapi karena tidak ada data untuk "
            "dinilai. Baca\n"
            "  `fold_class_coverage.csv` sebelum mengutip "
            "angka mana pun yang\n"
            "  berasal dari fold-fold ini."
        )

    print("=" * 68)

    return missing


def check_fold_integrity(
    labels_arr,
    skf=None,
    n_splits=N_SPLITS,
    verbose=True,
):

    """
    Dipanggil sekali di awal main() sebelum training dimulai.

    Mengembalikan DataFrame cakupan (fold x kelas) dan, kalau
    verbose=True, mencetak tabel peringatan (fold, kelas) yang
    kosong. Fungsi ini tidak pernah menghentikan pipeline.
    """

    coverage_df = build_fold_coverage_table(
        labels_arr,
        skf=skf,
        n_splits=n_splits,
    )

    if verbose:

        print_fold_integrity_warning(
            coverage_df,
            n_splits=n_splits,
        )

    return coverage_df


def per_class_metrics(labels, preds):
    """
    Precision/recall/F1/support per kelas (semua 8 kelas).

    Dipakai oleh compute_metrics supaya kelas langka terlihat,
    bukan tercebur di dalam weighted average.
    """

    label_ids = list(range(len(LABEL_MAP)))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            labels=label_ids,
            zero_division=0,
        )
    )

    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=[
            ID_TO_LABEL[i] for i in label_ids
        ],
        zero_division=0,
        output_dict=True,
    )

    out = {}

    for pos, label_id in enumerate(label_ids):

        name = ID_TO_LABEL[label_id]

        out[f"precision_{name}"] = float(
            precision[pos]
        )
        out[f"recall_{name}"] = float(
            recall[pos]
        )
        out[f"f1_{name}"] = float(f1[pos])
        out[f"support_{name}"] = int(
            support[pos]
        )

    out["macro_precision"] = float(
        np.mean(precision)
    )
    out["macro_recall"] = float(
        np.mean(recall)
    )
    out["macro_f1"] = float(
        np.mean(f1)
    )

    return out, report


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

    class_metrics, report = per_class_metrics(
        labels,
        preds,
    )

    # Trainer mengevaluasi tiap epoch pada set validasi yang sama,
    # jadi cetak laporan hanya sekali per komposisi fold.
    signature = tuple(
        sorted(
            (
                ID_TO_LABEL[i],
                int(n),
            )
            for i, n in enumerate(
                np.bincount(
                    np.asarray(labels),
                    minlength=len(LABEL_MAP),
                )
            )
        )
    )

    if signature not in _REPORTED_FOLD_SIGNATURES:

        _REPORTED_FOLD_SIGNATURES.add(signature)

        print(
            "\nClassification report "
            "(set validasi fold ini):"
        )

        print(
            classification_report(
                labels,
                preds,
                labels=list(range(len(LABEL_MAP))),
                target_names=[
                    ID_TO_LABEL[i]
                    for i in range(len(LABEL_MAP))
                ],
                zero_division=0,
                digits=3,
            )
        )

        absent = [
            name
            for name in report
            if isinstance(report[name], dict)
            and report[name]["support"] == 0
        ]

        if absent:

            print(
                f"[PERINGATAN] {len(absent)} kelas tidak ada "
                f"di set validasi ini: {absent}\n"
                "  Metrik kelas tersebut 0.000 karena "
                "support 0, bukan karena prediksi salah."
            )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **class_metrics,
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

        sparse_classes = get_sparse_classes(
            label_counts,
            n_splits=N_SPLITS,
        )

        print_sparse_class_warning(
            sparse_classes,
            n_rows=len(df),
            n_splits=N_SPLITS,
        )

    else:

        sparse_classes = []

    # Ekspos ke modul lain supaya tidak perlu menghitung ulang.
    global SPARSE_CLASSES

    SPARSE_CLASSES = [
        name for name, _ in sparse_classes
    ]

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

    # Deteksi (fold, kelas) yang tidak punya contoh validasi
    # SEBELUM training berjalan. Tidak menghentikan apa pun.
    coverage_df = check_fold_integrity(
        labels_arr,
        skf=skf,
        n_splits=N_SPLITS,
    )

    coverage_path = (
        CV_RESULT_DIR / "fold_class_coverage.csv"
    )

    CV_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    coverage_df.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nCakupan kelas per fold disimpan di: {coverage_path}"
    )

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