"""
Laporan ketidakseimbangan kelas pada 10-fold cross-validation (BUG-08).

APA MASALAHNYA
-------------
Training memakai StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
atas 800 baris berlabel 8 kelas.Empat kelas negatif besar, empat kelas
positif sangat kecil:

    akuntabilitas_negatif               305
    efektivitas_efisiensi_negatif      251
    responsivitas_negatif               92
    transparansi_negatif                69
    akuntabilitas_positif                58
    efektivitas_efisiensi_positif        12
    responsivitas_positif                 7
    transparansi_positif                   6

StratifiedKFold hanya bisa menugaskan minimal satu anggota kelas ke
setiap fold. Kelas dengan kurang dari 10 anggota karena itu PASTI
absen dari sebagian fold, dan di fold itu precision/recall/F1-nya 0.000
karena support 0 -- bukan karena prediksinya salah. Rata-rata berbobot
(walau f1 = 0.7609) tetap dihitung tanpa error, hanya saja kelas yang
tidak terlihat sama sekali ikut tersembunyi di dalamnya.

APA YANG SCRIPT INI LAKUKAN
---------------------------
Tanpa training ulang. Semua dihitung ulang dari artefak yang SUDAH ADA:

  1. Membaca `cv_per_fold_metrics.csv` + `cv_summary_metrics.csv`.
  2. Membangun ulang partisi StratifiedKFold(10, 42) yang persis sama
     dengan yang dipakai training, dengan mengimpor `load_and_clean_data`
     dari `indobert.train_cv`.
  3. Menghitung jumlah contoh validasi tiap kelas di tiap fold, lalu
     menandai pasangan (fold, kelas) yang bernilai 0.
  4. Menjumlahkan 10 confusion matrix yang sudah tersimpan menjadi satu
     confusion matrix global (800 pengujian, setiap baris hanya diuji
     sekali karena tiap baris masuk tepat satu fold), lalu menurunkan
     metrik per kelas darinya.

OUTPUT
------
  data/results/cv/per_class_metrics.csv    metrik per kelas
  data/results/cv/fold_class_coverage.csv  cakupan kelas x fold
  data/results/cv/confusion_matrix_overall.csv

Run:  python -m analysis.class_imbalance_report
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

import config  # noqa: E402

from indobert.train_cv import (  # noqa: E402
    ID_TO_LABEL,
    LABEL_MAP,
    N_SPLITS,
    RANDOM_STATE,
    load_and_clean_data,
)

warnings.filterwarnings(
    "ignore",
    message=".*least populated class.*",
)

# ── PATH ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

CV_DIR = BASE_DIR / config.CV_RESULTS_DIR

PER_FOLD_CSV = CV_DIR / "cv_per_fold_metrics.csv"
SUMMARY_CSV = CV_DIR / "cv_summary_metrics.csv"
CM_GLOB = f"confusion_matrix_fold_*.csv"

PER_CLASS_OUT = CV_DIR / "per_class_metrics.csv"
COVERAGE_OUT = CV_DIR / "fold_class_coverage.csv"
CM_OVERALL_OUT = CV_DIR / "confusion_matrix_overall.csv"

LINE = "=" * 74


def load_existing_metrics():
    """Baca artefak CV yang sudah ada. Tidak ada file -> berhenti jelas."""

    for path in (PER_FOLD_CSV, SUMMARY_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} tidak ditemukan. Jalankan "
                "`python -m indobert.train_cv` terlebih dahulu."
            )

    per_fold = pd.read_csv(PER_FOLD_CSV)
    summary = pd.read_csv(SUMMARY_CSV, index_col=0)

    return per_fold, summary


def load_confusion_matrices():
    """Baca seluruh confusion matrix per fold yang tersimpan."""

    paths = sorted(
        CV_DIR.glob(CM_GLOB),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    if not paths:
        raise FileNotFoundError(
            f"Tidak ada {CM_GLOB} di {CV_DIR}. "
            "Jalankan `python -m indobert.train_cv`."
        )

    label_order = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]

    frames = []

    for path in paths:
        fold_idx = int(path.stem.split("_")[-1])
        cm = pd.read_csv(path, index_col=0)
        cm = cm.loc[label_order, label_order]
        frames.append((fold_idx, cm))

    return paths, frames, label_order


def derive_split():
    """
    Bangun ulang partisi validasi yang sama persis dengan training.

    `load_and_clean_data` diimpor dari `indobert.train_cv` agar urutan
    baris dan pemetaan label tidak bisa melenceng dari partisi yang
    menghasilkan 10 checkpoint di models/.
    """

    df = load_and_clean_data()

    texts = df["clean_text"].astype(str).tolist()
    labels = df["label"].tolist()

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels)

    splits = list(
        skf.split(texts_arr, labels_arr)
    )

    return df, labels_arr, splits


def build_coverage(labels_arr, splits):
    """Jumlah contoh validasi tiap (fold, kelas)."""

    n_classes = len(LABEL_MAP)

    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):

        val_counts = np.bincount(
            labels_arr[val_idx],
            minlength=n_classes,
        )
        train_counts = np.bincount(
            labels_arr[train_idx],
            minlength=n_classes,
        )

        for label_id in range(n_classes):
            rows.append(
                {
                    "fold": fold_idx,
                    "label_id": label_id,
                    "label": ID_TO_LABEL[label_id],
                    "n_train": int(train_counts[label_id]),
                    "n_val": int(val_counts[label_id]),
                }
            )

    coverage = pd.DataFrame(rows)

    # KFold valid: jumlah baris = jumlah data, tidak ada tumpang tindih.
    total_val = int(coverage["n_val"].sum())
    assert total_val == len(labels_arr), (
        f"Partisi tidak konsisten: jumlah baris validasi "
        f"{total_val} != {len(labels_arr)} baris data."
    )
    assert coverage.groupby("fold")["n_val"].sum().eq(
        len(labels_arr) // N_SPLITS
    ).all(), "Ukuran fold tidak seragam."

    return coverage


def expand_confusion(overall_cm, label_order):
    """
    Bangun ulang pasangan (y_true, y_pred) dari confusion matrix global.

    Metrik apa pun hanya bergantung pada confusion matrix, jadi ini
    ekuivalen dengan menggabungkan prediksi out-of-fold asli: setiap
    baris anotasi masuk tepat satu fold validasi, sehingga totalnya
    tepat 800 dan tidak ada baris yang dihitung dua kali.
    """

    cm = overall_cm.to_numpy()

    y_true = []
    y_pred = []

    for row_pos in range(len(label_order)):
        for col_pos in range(len(label_order)):
            n = int(cm[row_pos, col_pos])
            if n:
                y_true.extend([row_pos] * n)
                y_pred.extend([col_pos] * n)

    return np.array(y_true), np.array(y_pred)


def per_class_from_confusion(overall_cm, label_order):
    """
    Turunkan metrik per kelas dari confusion matrix global.

    Baris = label asli, kolom = label prediksi (diformat persis seperti
    yang ditulis `run_fold`).
    """

    from sklearn.metrics import (
        precision_recall_fscore_support,
    )

    y_true, y_pred = expand_confusion(
        overall_cm,
        label_order,
    )

    label_ids = [LABEL_MAP[name] for name in label_order]

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            labels=label_ids,
            zero_division=0,
        )
    )

    cm = overall_cm.to_numpy()

    rows = []

    for pos, name in enumerate(label_order):
        tp = int(cm[pos, pos])
        fp = int(cm[:, pos].sum() - tp)
        fn = int(cm[pos, :].sum() - tp)
        sup = int(cm[pos, :].sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1c = (
            2 * prec * rec / (prec + rec)
            if (prec + rec)
            else 0.0
        )

        # Angka manual harus sama persis dengan sklearn; kalau tidak,
        # ada bug di pembacaan confusion matrix.
        assert abs(prec - float(precision[pos])) < 1e-9
        assert abs(rec - float(recall[pos])) < 1e-9
        assert abs(f1c - float(f1[pos])) < 1e-9

        rows.append(
            {
                "label": name,
                "label_id": LABEL_MAP[name],
                "support": sup,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1c, 4),
            }
        )

    return pd.DataFrame(rows)


def print_header():
    print(LINE)
    print("LAPORAN KETIMPANGAN KELAS -- 10-FOLD CROSS VALIDATION (BUG-08)")
    print(LINE)


def main():
    print_header()

    # ── 1. artefak CV yang sudah ada ──────────────────────────────────────
    per_fold, summary = load_existing_metrics()
    cm_paths, cm_frames, label_order = load_confusion_matrices()

    print(f"\nArtefak yang dibaca (tanpa training ulang):")
    print(f"  {PER_FOLD_CSV.relative_to(BASE_DIR)}  ({len(per_fold)} fold)")
    print(f"  {SUMMARY_CSV.relative_to(BASE_DIR)}")
    print(f"  {len(cm_paths)} confusion matrix per fold")

    print("\nRingkasan yang dilaporkan training (dari cv_summary_metrics.csv):")
    for metric in ("eval_accuracy", "eval_precision", "eval_recall", "eval_f1"):
        if metric in summary.index:
            row = summary.loc[metric]
            print(
                f"  {metric:<16} mean {float(row['mean']):.4f}   "
                f"std {float(row['std']):.4f}"
            )

    print(
        "\nAngka-angka ini memakai weighted average, jadi kelas dengan "
        "support 0\ntidak terlihat. Bagian di bawah menguraikannya."
    )

    # ── 2. partisi validasi yang sama ─────────────────────────────────────
    df, labels_arr, splits = derive_split()

    print(f"\nPartisi StratifiedKFold({N_SPLITS}, shuffle=True, "
          f"random_state={RANDOM_STATE})")
    sizes = [len(v) for _, v in splits]
    print(
        f"  {len(splits)} fold x {min(sizes)}-{max(sizes)} baris "
        f"= {sum(sizes)} baris (data: {len(df)})"
    )

    coverage = build_coverage(labels_arr, splits)

    # Bukti bahwa partisi yang dibangun ulang ini MEMANG partisi yang
    # menghasilkan checkpoint: jumlah baris asli tiap kelas di setiap
    # confusion matrix tersimpan harus sama persis dengan n_val.
    mismatches = []

    for fold_idx, cm in cm_frames:
        derived = (
            coverage[coverage["fold"] == fold_idx]
            .set_index("label_id")["n_val"]
            .to_numpy()
        )
        from_cm = (
            cm.to_numpy().sum(axis=1)
        )

        if not (derived == from_cm).all():
            mismatches.append(fold_idx)

    print(
        "  Cocok dengan confusion matrix per fold yang tersimpan: "
        + ("YA (10/10)" if not mismatches
           else f"TIDAK (fold {mismatches})")
    )

    # ── 3. ringkasan cakupan per kelas ────────────────────────────────────
    per_class_coverage = []

    for name in label_order:
        sub = coverage[coverage["label"] == name]
        n_val = sub["n_val"]
        zero_folds = sorted(
            sub.loc[sub["n_val"] == 0, "fold"].tolist()
        )
        per_class_coverage.append(
            {
                "label": name,
                "label_id": LABEL_MAP[name],
                "total": int(n_val.sum()),
                "min_per_fold": int(n_val.min()),
                "max_per_fold": int(n_val.max()),
                "n_folds_zero": len(zero_folds),
                "folds_zero": ";".join(str(f) for f in zero_folds),
                "sparse": len(zero_folds) > 0,
            }
        )

    cov_df = pd.DataFrame(per_class_coverage)

    print("\n" + LINE)
    print("A. CAKUPAN KELAS DI SEPULUH FOLD VALIDASI")
    print(LINE)
    print(
        f"{'label':<32}{'total':>7}{'min':>6}{'max':>6}"
        f"{'fold kosong':>13}  fold"
    )
    for _, row in cov_df.iterrows():
        zeros = row["folds_zero"] or "-"
        print(
            f"{row['label']:<32}{int(row['total']):>7}"
            f"{int(row['min_per_fold']):>6}"
            f"{int(row['max_per_fold']):>6}"
            f"{int(row['n_folds_zero']):>13}  {zeros}"
        )

    zero_pairs = coverage[coverage["n_val"] == 0]
    print(f"\nTotal pasangan (fold, kelas) tanpa contoh validasi: "
          f"{len(zero_pairs)}")

    if not zero_pairs.empty:
        print(f"\n{'fold':>6}{'label':>34}{'n_val':>7}")
        for _, row in zero_pairs.sort_values(["fold", "label_id"]).iterrows():
            print(
                f"{int(row['fold']):>6}{row['label']:>34}"
                f"{int(row['n_val']):>7}"
            )

    # ── 4. metrik per kelas dari confusion matrix ─────────────────────────
    overall_cm = cm_frames[0][1].copy()
    for _, cm in cm_frames[1:]:
        overall_cm = overall_cm.add(cm, fill_value=0)

    total_cells = int(overall_cm.to_numpy().sum())
    print(f"\nJumlah 10 confusion matrix -> {total_cells} prediksi "
          f"(target {len(df)})")

    per_class = per_class_from_confusion(overall_cm, label_order)

    # gabungkan cakupan fold ke tabel metrik
    per_class = per_class.merge(
        cov_df[[
            "label",
            "total",
            "min_per_fold",
            "max_per_fold",
            "n_folds_zero",
            "folds_zero",
        ]],
        on="label",
    )

    per_class = per_class.sort_values(
        ["support", "label_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    weighted_f1 = float(
        np.average(
            per_class["f1"],
            weights=per_class["support"],
        )
    )
    weighted_p = float(
        np.average(
            per_class["precision"],
            weights=per_class["support"],
        )
    )
    weighted_r = float(
        np.average(
            per_class["recall"],
            weights=per_class["support"],
        )
    )
    macro_f1 = float(per_class["f1"].mean())
    accuracy = float(
        np.trace(overall_cm.to_numpy()) / total_cells
    )

    print("\n" + LINE)
    print("B. METRIK PER KELAS (dari 10 confusion matrix yang "
          "sudah ada)")
    print(LINE)
    print(
        f"{'label':<32}{'supp':>6}{'P':>8}{'R':>8}{'F1':>8}"
        f"{'fold kosong':>13}"
    )
    for _, row in per_class.iterrows():
        print(
            f"{row['label']:<32}"
            f"{int(row['support']):>6}"
            f"{row['precision']:>8.4f}"
            f"{row['recall']:>8.4f}"
            f"{row['f1']:>8.4f}"
            f"{int(row['n_folds_zero']):>13}"
        )

    print(f"\n  accuracy (8 kelas, 800 baris) : {accuracy:.4f}")
    print(f"  weighted precision            : {weighted_p:.4f}")
    print(f"  weighted recall               : {weighted_r:.4f}")
    print(f"  weighted F1                   : {weighted_f1:.4f}")
    print(f"  macro F1                      : {macro_f1:.4f}")

    report = summary.loc["eval_accuracy", "mean"] if (
        "eval_accuracy" in summary.index
    ) else np.nan
    report_f1 = summary.loc["eval_f1", "mean"] if (
        "eval_f1" in summary.index
    ) else np.nan

    print(
        f"\n  /check: rerata akurasi per fold yang dilaporkan "
        f"({float(report):.4f})\n"
        f"  vs akurasi global dari confusion matrix "
        f"({accuracy:.4f})\n"
        f"  /check: rerata weighted F1 per fold yang dilaporkan "
        f"({float(report_f1):.4f})\n"
        f"  vs weighted F1 global dari confusion matrix "
        f"({weighted_f1:.4f})"
    )

    print(
        "\n  Catatan: akurasi cocok PERSIS karena setiap fold sama "
        "besar (80),\n  jadi rata-rata per fold = nilai global. "
        "F1 dan precision tidak\n  cocok persis karena `cv_summary_"
        "metrics.csv` berisi rata-rata\n  dari 10 fold, sedangkan "
        "angka di atas dihitung sekali atas\n  800 prediksi gabungan. "
        "Keduanya sah; untuk makalah,\n  angka global di sini lebih "
        "konservatif dan tidak bergantung pada\n  bagaimana kelas "
        "langka jatuh di fold tertentu."
    )

    # ── 5. kelas yang tidak bisa dinilai ─────────────────────────────────
    unreliable = per_class[
        per_class["n_folds_zero"] > 0
    ]
    dead = per_class[per_class["tp"] == 0]

    print("\n" + LINE)
    print("C. KELAS YANG GANGGU DIPERCAYAI")
    print(LINE)
    print("1. Kelas dengan 0 contoh di >=1 fold. Metrik per kelasnya "
          "dihitung dari\n   fold yang benar-benar punya contoh saja, "
          "jadi angkanya di sini\n   lebih tinggi daripada yang "
          "dilihat fold per fold.")
    if not unreliable.empty:
        for _, row in unreliable.iterrows():
            print(
                f"   - {row['label']}: {int(row['n_folds_zero'])}/"
                f"{N_SPLITS} fold kosong (fold "
                f"{row['folds_zero']}), support total "
                f"{int(row['support'])}"
            )
    print("\n2. Kelas dengan support kecil. Satu kesalahan saja "
          "menggeser recall\n   jauh lebih besar daripada pada "
          "kelas besar.")
    tiny = per_class[per_class["support"] < N_SPLITS * 3]
    for _, row in tiny.iterrows():
        print(
            f"   - {row['label']}: support {int(row['support'])}, "
            f"P {row['precision']:.3f} / R {row['recall']:.3f} / "
            f"F1 {row['f1']:.3f}"
        )
    print("\n3. Kelas yang tidak pernah diprediksi benar sama sekali "
          "(TP = 0):")
    if dead.empty:
        print("   (tidak ada)")
    else:
        for _, row in dead.iterrows():
            print(
                f"   - {row['label']}: support {int(row['support'])}, "
                f"semua {int(row['fn'])} contoh diprediksi sebagai "
                "kelas lain"
            )

    print("\n" + LINE)
    print("D. REKOMENDASI")
    print(LINE)
    min_support = int(per_class["support"].min())
    n_sparse = int((per_class["n_folds_zero"] > 0).sum())
    print(
        f"Kelas terkecil punya {min_support} contoh. Dengan "
        f"n_splits={N_SPLITS},\n{n_sparse} kelas tidak pernah "
        "muncul di semua fold, sehingga angka\nweighted F1 "
        f"({weighted_f1:.4f}) meremehkan keterbatasan model "
        f"justru di {n_sparse} kelas\npositif yang paling langka."
    )
    print(
        "\nOpsi, dari yang paling murah ke yang paling invasive:\n"
        "  A. Pertahankan n_splits=10, laporkan metrik per kelas di "
        "atas, dan\n     tulis caveat eksplisit di Bab 4. "
        "Biaya: nol. Ini yang\n     direkomendasikan untuk "
        "skripsi yang deadline-nya dekat.\n"
        f"  B. Turunkan n_splits ke <= {min_support} agar setiap "
        "kelas muncul di\n     semua fold. Biaya: training ulang "
        "total, dan kekuatan kontrol\n     eksperimen melemah dari "
        f"{N_SPLITS} fit menjadi {min_support} fit.\n"
        "  C. Gabungkan *_positif yang sangat langka menjadi satu "
        "kelas\n     `positif_lainnya`, atau turunkan masalah "
        "positive menjadi\n     kelas binary pos/neg per aspek. "
        "Perubahan kelas menyentuh\n     annotation, semua "
        "checkpoint, dan semua script analisis -- besar.\n"
        "  D. Tambah data anotasi untuk Responsivitas/Transparansi "
        "positif.\n     Satu-satunya opsi yang benar-benar "
        "menghilangkan keterbatasan;\n     butuh usaha "
        "anotasi, bukan kode."
    )
    print(
        "\nYang TIDAK_BOLEH dilakukan: mengutip "
        f"{weighted_f1:.4f} sebagai bukti\nkelas positif "
        "terdeteksi. Angka itu benar sebagai rata-rata "
        "berbobot,\ntapi ia menutupi kelas dengan F1 0.000. "
        "Gunakan tabel bagian B."
    )

    # ── 6. simpan ─────────────────────────────────────────────────────────
    CV_DIR.mkdir(parents=True, exist_ok=True)

    per_class.to_csv(
        PER_CLASS_OUT,
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        COVERAGE_OUT,
        index=False,
        encoding="utf-8-sig",
    )
    overall_cm.to_csv(
        CM_OVERALL_OUT,
        encoding="utf-8-sig",
    )

    print(f"\nTersimpan:")
    print(f"  {PER_CLASS_OUT.relative_to(BASE_DIR)}")
    print(f"  {COVERAGE_OUT.relative_to(BASE_DIR)}")
    print(f"  {CM_OVERALL_OUT.relative_to(BASE_DIR)}")
    print(LINE)


if __name__ == "__main__":
    main()
