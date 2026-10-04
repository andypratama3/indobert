import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import config

# ======================================================
# PATH
# ======================================================

INPUT_PATH = config.ABSA_RESULT

OUTPUT_DIR = Path(
    "data/results/visualization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ======================================================
# DISTRIBUSI ASPEK
# ======================================================

def plot_aspect_distribution(df):

    counts = (
        df["predicted_aspect"]
        .value_counts()
    )

    plt.figure(figsize=(8,5))

    ax = counts.plot(
        kind="bar"
    )

    plt.title(
        "Distribusi Aspek Hasil Prediksi",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Aspek")
    plt.ylabel("Jumlah Komentar")

    plt.xticks(rotation=15)

    for p in ax.patches:

        ax.annotate(
            str(int(p.get_height())),
            (
                p.get_x() + p.get_width()/2,
                p.get_height()
            ),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "distribusi_aspek.png",
        dpi=300
    )

    plt.close()


# ======================================================
# DISTRIBUSI SENTIMEN
# ======================================================

def plot_sentiment_distribution(df):

    counts = (
        df["predicted_sentiment"]
        .value_counts()
    )

    plt.figure(figsize=(6,5))

    ax = counts.plot(
        kind="bar"
    )

    plt.title(
        "Distribusi Sentimen Hasil Prediksi",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah Komentar")

    for p in ax.patches:

        ax.annotate(
            str(int(p.get_height())),
            (
                p.get_x() + p.get_width()/2,
                p.get_height()
            ),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "distribusi_sentimen.png",
        dpi=300
    )

    plt.close()


# ======================================================
# DISTRIBUSI ASPEK DAN SENTIMEN
# ======================================================

def plot_aspect_sentiment(df):

    cross = pd.crosstab(
        df["predicted_aspect"],
        df["predicted_sentiment"]
    )

    ax = cross.plot(
        kind="bar",
        stacked=True,
        figsize=(9,6)
    )

    plt.title(
        "Distribusi Aspek dan Sentimen Hasil Prediksi",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Aspek")
    plt.ylabel("Jumlah Komentar")

    plt.xticks(rotation=15)

    plt.legend(
        title="Sentimen"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "aspect_sentiment_distribution.png",
        dpi=300
    )

    plt.close()


# ======================================================
# SUMMARY
# ======================================================

def save_summary(df):

    aspect = (
        df["predicted_aspect"]
        .value_counts()
        .reset_index()
    )

    aspect.columns = [
        "Aspek",
        "Jumlah"
    ]

    aspect["Persentase"] = (
        aspect["Jumlah"]
        / aspect["Jumlah"].sum()
        *100
    ).round(2)

    aspect.to_csv(
        OUTPUT_DIR /
        "summary_aspek.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sentiment = (
        df["predicted_sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment.columns = [
        "Sentimen",
        "Jumlah"
    ]

    sentiment["Persentase"] = (
        sentiment["Jumlah"]
        / sentiment["Jumlah"].sum()
        *100
    ).round(2)

    sentiment.to_csv(
        OUTPUT_DIR /
        "summary_sentimen.csv",
        index=False,
        encoding="utf-8-sig"
    )

    aspect_sentiment = pd.crosstab(
        df["predicted_aspect"],
        df["predicted_sentiment"]
    )

    aspect_sentiment.to_csv(
        OUTPUT_DIR /
        "summary_aspect_sentiment.csv",
        encoding="utf-8-sig"
    )


# ======================================================
# MAIN
# ======================================================

def main():

    print("="*60)
    print("VISUALISASI HASIL ANALISIS SENTIMEN BERBASIS ASPEK")
    print("="*60)

    df = pd.read_csv(
        INPUT_PATH
    )

    plot_aspect_distribution(df)

    plot_sentiment_distribution(df)

    plot_aspect_sentiment(df)

    save_summary(df)

    print("\nVisualisasi berhasil dibuat.")

    print(
        f"\nJumlah komentar : {len(df)}"
    )

    print(
        f"\nHasil disimpan di:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()