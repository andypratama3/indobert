"""
Pipeline: Identifikasi Permasalahan + RCA + Rekomendasi
Input  : data/results/indobert_absa_result_oof.csv  (leakage-free predictions)
Output : data/results/identifikasi_permasalahan/

Note: previously read data/results/indobert_absa_result.csv, which contained
out-of-fold-contaminated predictions. See docs/BUG_TRACKER.md BUG-04.
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")

# ── PATH ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = str(config.BASE_DIR / config.ABSA_RESULT)
OUT_DIR   = str(config.BASE_DIR / config.IDENTIFIKASI_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE   = 42     # seed tetap agar sampling dapat direproduksi
N_SAMPLE_MAKS  = 8      # jumlah maksimal sampel komentar per aspek yang ditampilkan

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA ...")
df = pd.read_csv(INPUT_CSV)
print(f"  Total komentar : {len(df)}")

ASPEK_LIST = [
    "Akuntabilitas",
    "Efektivitas dan Efisiensi",
    "Responsivitas",
    "Transparansi",
]

# ── STEP 1: DISTRIBUSI ASPEK DAN SENTIMEN ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — DISTRIBUSI ASPEK DAN SENTIMEN ...")

distribusi_rows = []
for aspek in ASPEK_LIST:
    sub     = df[df["predicted_aspect"] == aspek]
    total   = len(sub)
    negatif = len(sub[sub["predicted_sentiment"] == "Negatif"])
    positif = len(sub[sub["predicted_sentiment"] == "Positif"])
    pct_neg = round(negatif / total * 100, 2) if total > 0 else 0
    pct_pos = round(positif / total * 100, 2) if total > 0 else 0
    pct_vol = round(total / len(df) * 100, 2)
    distribusi_rows.append({
        "aspek": aspek,
        "total_komentar": total,
        "pct_dari_total": pct_vol,
        "jumlah_negatif": negatif,
        "pct_negatif": pct_neg,
        "jumlah_positif": positif,
        "pct_positif": pct_pos,
    })

df_distribusi = (pd.DataFrame(distribusi_rows)
                  .sort_values("jumlah_negatif", ascending=False)
                  .reset_index(drop=True))
df_distribusi.to_csv(f"{OUT_DIR}/distribusi_aspek_sentimen.csv", index=False, encoding="utf-8-sig")
print(df_distribusi.to_string(index=False))
print(f"\n  Saved: distribusi_aspek_sentimen.csv")

# ── STEP 2: TELAAH KUALITATIF (SAMPLING KOMENTAR NEGATIF) ────────────────────
print("\n" + "=" * 60)
print("STEP 2 — TELAAH KUALITATIF KOMENTAR NEGATIF ...")

sampel_rows = []
for aspek in ASPEK_LIST:
    subset = df[
        (df["predicted_aspect"] == aspek) &
        (df["predicted_sentiment"] == "Negatif")
    ]
    n_ambil = min(N_SAMPLE_MAKS, len(subset))
    if n_ambil == 0:
        continue
    sampel = subset.sample(n=n_ambil, random_state=RANDOM_STATE)
    for _, row in sampel.iterrows():
        sampel_rows.append({
            "aspek": aspek,
            "komentar": row["clean_text"],
            "aspek_prediksi": row["predicted_aspect"],
            "sentimen_prediksi": row["predicted_sentiment"],
        })
    print(f"  [{aspek}] diambil {n_ambil} sampel dari {len(subset)} komentar negatif")

df_sampel = pd.DataFrame(sampel_rows)
df_sampel.to_csv(f"{OUT_DIR}/sampel_komentar_negatif.csv", index=False, encoding="utf-8-sig")
print(f"\n  Saved: sampel_komentar_negatif.csv")
print("  Catatan: tema/pola keluhan dirangkum secara manual oleh peneliti")
print("  berdasarkan pembacaan sampel ini (lihat RCA_DATA pada Step 3).")

# ── STEP 3: RCA (5 WHYS) + REKOMENDASI ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — RCA (5 WHYS) + REKOMENDASI ...")

# Rangkuman tema/pola keluhan dan hasil 5 Whys disusun secara kualitatif oleh
# peneliti berdasarkan distribusi (Step 1) dan telaah sampel komentar (Step 2).
RCA_DATA = {
    "Akuntabilitas": {
        "isu": "Pertanggungjawaban penggunaan anggaran Rp 8,5 miliar untuk pengadaan mobil dinas dipertanyakan publik",
        "why1": "Masyarakat tidak mengetahui detail dan justifikasi anggaran yang digunakan",
        "why2": "Rincian pengadaan tidak dipublikasikan secara proaktif kepada publik",
        "why3": "Tidak ada mekanisme publikasi wajib rincian pengadaan ke publik luas",
        "why4": "Mekanisme akuntabilitas publik dalam proses pengadaan barang daerah masih lemah",
        "why5": "Belum ada sistem pelaporan pengadaan yang terintegrasi dan mudah diakses masyarakat",
        "akar_masalah": "Lemahnya mekanisme akuntabilitas publik dalam proses pengadaan barang pemerintah daerah",
        "rekomendasi": "1. Publikasikan rincian kebutuhan operasional, spesifikasi teknis, dan dasar penetapan harga pengadaan. 2. Terapkan mekanisme pelaporan pengadaan yang transparan. 3. Libatkan DPRD dan masyarakat dalam pembahasan anggaran pengadaan.",
    },
    "Efektivitas dan Efisiensi": {
        "isu": "Masyarakat menilai anggaran Rp 8,5 miliar tidak tepat sasaran dan lebih mendesak untuk infrastruktur jalan",
        "why1": "Masyarakat melihat jalan rusak di Kalimantan Timur namun pemerintah mengalokasikan anggaran besar untuk kendaraan dinas",
        "why2": "Tidak ada penjelasan resmi tentang urgensi dan skala prioritas pengadaan mobil dinas",
        "why3": "Pemerintah tidak mengomunikasikan skala prioritas belanja daerah secara proaktif",
        "why4": "Tidak ada mekanisme komunikasi kebijakan anggaran yang sistematis ke masyarakat",
        "why5": "Belum ada sistem monitoring opini publik untuk mengantisipasi respons masyarakat sebelum kebijakan diimplementasikan",
        "akar_masalah": "Tidak adanya mekanisme komunikasi proaktif tentang skala prioritas belanja daerah kepada publik",
        "rekomendasi": "1. Publikasikan skala prioritas belanja daerah secara terbuka. 2. Tunjukkan roadmap perbaikan infrastruktur jalan yang sudah direncanakan. 3. Lakukan sosialisasi kebijakan pengadaan sebelum implementasi.",
    },
    "Responsivitas": {
        "isu": "Pemerintah dinilai tidak memberikan respons yang memadai terhadap kritik publik terkait kebijakan pengadaan",
        "why1": "Tidak ada pernyataan resmi pemerintah yang menjawab kritik masyarakat secara substantif",
        "why2": "Tidak ada protokol komunikasi krisis untuk merespons kontroversi kebijakan di media sosial",
        "why3": "Pemerintah belum mengintegrasikan media sosial sebagai kanal komunikasi kebijakan yang aktif",
        "why4": "Tidak ada tim khusus yang bertugas memantau dan merespons opini publik digital",
        "why5": "Belum ada sistem monitoring media sosial dan protokol respons cepat di pemerintah daerah Kalimantan Timur",
        "akar_masalah": "Belum adanya protokol komunikasi krisis berbasis media sosial dan sistem monitoring opini publik digital di pemerintah daerah",
        "rekomendasi": "1. Bentuk tim komunikasi publik yang memantau dan merespons opini masyarakat di media sosial. 2. Buat pernyataan resmi yang menjawab pertanyaan publik secara substantif. 3. Manfaatkan kanal media sosial resmi pemerintah untuk komunikasi kebijakan secara proaktif.",
    },
    "Transparansi": {
        "isu": "Masyarakat merasa tidak mendapatkan informasi yang cukup mengenai proses dan dasar keputusan pengadaan mobil dinas",
        "why1": "Informasi tentang proses pengadaan tidak tersedia secara mudah di kanal publik",
        "why2": "Tidak ada inisiatif keterbukaan informasi pengadaan dari pemerintah secara proaktif",
        "why3": "Mekanisme keterbukaan informasi publik untuk pengadaan barang daerah belum berjalan optimal",
        "why4": "Kesadaran dan komitmen penerapan prinsip transparansi dalam pengadaan masih rendah",
        "why5": "Belum ada sistem informasi pengadaan yang terintegrasi dan dapat diakses publik secara real-time",
        "akar_masalah": "Belum optimalnya penerapan prinsip keterbukaan informasi publik dalam proses pengadaan barang pemerintah daerah",
        "rekomendasi": "1. Publikasikan dokumen pengadaan melalui portal LPSE atau website resmi pemerintah. 2. Buat kanal informasi khusus yang memudahkan masyarakat mengakses informasi pengadaan. 3. Terapkan prinsip open government dalam setiap proses pengadaan.",
    },
}

rca_rows = []
for _, row in df_distribusi.iterrows():
    aspek = row["aspek"]
    rca   = RCA_DATA.get(aspek, {})
    rca_rows.append({
        "aspek"          : aspek,
        "total_komentar" : row["total_komentar"],
        "pct_negatif"    : f"{row['pct_negatif']}%",
        "pct_positif"    : f"{row['pct_positif']}%",
        "isu_kritis"     : rca.get("isu", "-"),
        "why1"           : rca.get("why1", "-"),
        "why2"           : rca.get("why2", "-"),
        "why3"           : rca.get("why3", "-"),
        "why4"           : rca.get("why4", "-"),
        "why5"           : rca.get("why5", "-"),
        "akar_masalah"   : rca.get("akar_masalah", "-"),
        "rekomendasi"    : rca.get("rekomendasi", "-"),
    })

df_rca = pd.DataFrame(rca_rows)
df_rca.to_csv(f"{OUT_DIR}/rca_rekomendasi.csv", index=False, encoding="utf-8-sig")
print(f"  Saved: rca_rekomendasi.csv")

# ── STEP 4: CSV LAPORAN LENGKAP ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — CSV LAPORAN LENGKAP ...")

df_rca.to_csv(f"{OUT_DIR}/laporan_lengkap.csv", index=False, encoding="utf-8-sig")
print(f"  Saved: laporan_lengkap.csv")

# ── STEP 5: VISUALISASI PNG ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — VISUALISASI PNG ...")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

ASPEK_COLOR = {
    "Akuntabilitas"            : "#e74c3c",
    "Efektivitas dan Efisiensi": "#e67e22",
    "Responsivitas"            : "#9b59b6",
    "Transparansi"             : "#3498db",
}

# PNG 1: Distribusi Sentimen Negatif per Aspek (jumlah & persentase)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = [ASPEK_COLOR.get(a, "#95a5a6") for a in df_distribusi["aspek"]]

axes[0].barh(df_distribusi["aspek"], df_distribusi["jumlah_negatif"], color=colors, alpha=0.85)
axes[0].set_xlabel("Jumlah Komentar Negatif")
axes[0].set_title("Jumlah Komentar Negatif per Aspek", fontsize=11, fontweight="bold")
axes[0].invert_yaxis()
axes[0].grid(axis="x", linestyle="--", alpha=0.4)

axes[1].barh(df_distribusi["aspek"], df_distribusi["pct_negatif"], color=colors, alpha=0.85)
axes[1].set_xlabel("Persentase Sentimen Negatif (%)")
axes[1].set_title("Persentase Sentimen Negatif per Aspek", fontsize=11, fontweight="bold")
axes[1].invert_yaxis()
axes[1].set_xlim(0, 100)
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

fig.suptitle(
    "Distribusi Sentimen Negatif per Aspek Tata Kelola\nPengadaan Mobil Dinas Gubernur Kalimantan Timur",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/distribusi_negatif_per_aspek.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: distribusi_negatif_per_aspek.png")

# ── SELESAI ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE SELESAI! OUTPUT ADA DI:")
print(f"  data/results/identifikasi_permasalahan/")
print(f"  distribusi_aspek_sentimen.csv")
print(f"  sampel_komentar_negatif.csv   <- lampirkan di Bab 5 (komentar + aspek + sentimen)")
print(f"  rca_rekomendasi.csv")
print(f"  laporan_lengkap.csv           <- buka di Excel")
print(f"  distribusi_negatif_per_aspek.png")
print("=" * 60)