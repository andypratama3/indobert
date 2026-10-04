"""
plot_thematic_distribution.py

Membuat visualisasi Gambar 5.10 (Distribusi Tema Keluhan per Aspek), sebagai
pelengkap Tabel 5.15 pada subbab 5.8.3. Gaya visual disamakan dengan Gambar 5.9
(horizontal bar chart per aspek), tetapi di sini setiap aspek difasetkan
terpisah (small multiples) karena tema keluhannya berbeda-beda per aspek.

Input : data/results/thematic_summary.csv
        (dihasilkan oleh thematic_coding.py)
Output: data/results/visualization/gambar_5_10_distribusi_tema.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

INPUT_PATH = Path("data/results/thematic_summary.csv")
OUTPUT_PATH = Path("data/results/visualization/gambar_5_10_distribusi_tema.png")

# Kategori residual yang tidak ditampilkan di grafik (biar fokus ke tema utama,
# sama seperti yang ditampilkan pada Tabel 5.15)
EXCLUDE_LABEL = "Lainnya/tidak terklasifikasi"

# Urutan aspek mengikuti urutan pada Tabel 5.13 / Gambar 5.9
ASPEK_ORDER = [
    "Akuntabilitas",
    "Efektivitas dan Efisiensi",
    "Responsivitas",
    "Transparansi",
]

# Warna disamakan dengan skema Gambar 5.9 (biru, oranye, ungu, biru muda)
ASPEK_COLORS = {
    "Akuntabilitas": "#4C72B0",
    "Efektivitas dan Efisiensi": "#DD8452",
    "Responsivitas": "#8172B2",
    "Transparansi": "#64B5CD",
}


def wrap_label(text, width=32):
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df[df["tema"] != EXCLUDE_LABEL].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    fig.suptitle(
        "Distribusi Tema Keluhan pada Komentar Negatif per Aspek Tata Kelola",
        fontsize=18,
        fontweight="bold",
    )

    for ax, aspek in zip(axes.flatten(), ASPEK_ORDER):
        sub = (
            df[df["predicted_aspect"] == aspek]
            .sort_values("persentase (%)", ascending=True)
        )
        labels = [wrap_label(t, width=26) for t in sub["tema"]]
        values = sub["persentase (%)"]

        bars = ax.barh(labels, values, color=ASPEK_COLORS[aspek])
        ax.set_title(aspek, fontsize=15, fontweight="bold")
        ax.set_xlabel("Persentase terhadap total komentar negatif (%)", fontsize=12)
        ax.set_xlim(0, max(values) * 1.25)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=11)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[OK] Gambar disimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()