"""
thematic_coding.py

Tahap analisis lanjutan setelah prediksi ABSA (predict_absa.py).
Melakukan coding tematik (thematic analysis) terhadap komentar bersentimen
Negatif pada setiap aspek tata kelola, sebagai dasar penyusunan Tabel 5.13
dan Tabel 5.14 pada BAB 5.

Input : data/results/indobert_absa_result.csv
        (wajib punya kolom: clean_text, predicted_aspect, predicted_sentiment)
Output: data/results/thematic_coding_result.csv   -> data lengkap + kolom 'tema'
        data/results/thematic_summary.csv         -> rekap frekuensi tema per aspek
"""

import re
import pandas as pd
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# 1. Konfigurasi path (mengikuti struktur folder project)
# ---------------------------------------------------------------------------
INPUT_PATH = config.ABSA_RESULT
OUTPUT_DETAIL_PATH = Path("data/results/thematic_coding_result.csv")
OUTPUT_SUMMARY_PATH = Path("data/results/thematic_summary.csv")

# ---------------------------------------------------------------------------
# 2. Kamus tema (theme dictionary) per aspek
#    Disusun berdasarkan definisi operasional aspek pada hybrid annotation:
#      - Akuntabilitas          -> tanggung jawab & kinerja
#      - Efektivitas & Efisiensi-> manfaat kebijakan, ketepatan anggaran,
#                                   prioritas pembangunan, kinerja pemerintah
#      - Responsivitas          -> respon pemerintah terhadap aspirasi/kritik
#      - Transparansi           -> keterbukaan informasi, klarifikasi,
#                                   penjelasan, kejujuran pemerintah
#    Catatan: kamus ini adalah TITIK AWAL (hasil coding manual pada sampel).
#    Silakan disempurnakan lagi manual sebelum dipakai sebagai data final skripsi.
# ---------------------------------------------------------------------------
THEME_DICTIONARY = {
    "Akuntabilitas": {
        "Dugaan penyalahgunaan wewenang/KKN & tuntutan pemeriksaan": [
            "kpk", "korupsi", "kkn", "pidana", "periksa", "amplop coklat",
            "pemalsuan", "dijajah", "nggarong",
        ],
        "Kesenjangan gaya hidup mewah pejabat (termasuk keluarga) vs empati rakyat": [
            "marwah", "warwah", "mewah", "naik.*jelek", "bagus.*jelek",
            "ala kadarnya", "kasta", "sopir", "naik motor", "hedon",
            "mikir rakyat", "peduli rakyat", "istri", "fashion", "kebaya",
            "diana", "bridgerton", "raja", "over power",
        ],
        "Kinerja kepemimpinan dinilai buruk/mementingkan diri sendiri": [
            "pemimpin", "kerjaannya", "kerjanya", "tidak di pecat",
            "kepala daerah", "tidak becus", "zonk", "kinerja",
            "mementingkan diri", "tamak", "moral pejabat",
        ],
        "Tuntutan pertanggungjawaban politik (evaluasi/jangan dipilih lagi)": [
            "jangan dipilih", "tidak dipilih", "periode", "priode", "pecat",
            "copot", "evaluasi", "salah pilih", "nyesel.*pilih",
            "siapa yang pilih",
        ],
        "Sinisme terhadap kompetensi/kepantasan menjabat": [
            "kok bisa jadi gubernur", "iq gubernur", "kebingungan",
            "kok di sandingin", "beda kelas", "bingung.*gubernur",
        ],
        "Rasa malu warga & sorotan negatif nama daerah": [
            "malu", "dibully", "di bully", "korups", "amplop",
        ],
    },
    "Efektivitas dan Efisiensi": {
        "Prioritas anggaran salah (mobil vs infrastruktur jalan)": [
            "jalan", "jln", "aspal", "infrastruktur", "hotmix", "cor lebar",
        ],
        "Harga/nilai mobil & anggaran dianggap tidak wajar/boros": [
            "mahal", "miliar", "juta", "boros", "700jt", "700 jt", "8m",
            "9m", "8 5", "apbd", "pajak", "gengsi", "duit rakyat",
            "ngabisin", "mewah", "pajero", "hilux", "panther", "alphard",
            "alpard", "avanza", "triton", "defender", "raptor", "maung",
            "zenix", "rental", "pesawat", "dana pribadi",
        ],
    },
    "Responsivitas": {
        "Kritik respons pejabat pusat: sindir tanpa tindakan tegas": [
            "sindir", "tindak", "gosip", "omong", "ngomong", "tegur",
        ],
        "Minimnya suara/keterlibatan warga & pejabat daerah Kaltim": [
            "diam", "warga kaltim", "masyarakat kaltim", "abai",
        ],
    },
    "Transparansi": {
        "Kejanggalan identitas/legalitas kendaraan (plat nomor)": ["plat"],
        "Ketidakjujuran/kejanggalan narasi pengembalian mobil": [
            "kembali", "dikembalikan", "balikin", "dibalikin", "tipu",
            "jangal", "bodoh", "dijual",
        ],
        "Ketidakjelasan angka anggaran/dana (simpang siur nominal)": [
            "miliar", "juta", "anggaran", "8 5", "127", "137", "25m",
            "renovasi",
        ],
        "Minim klarifikasi resmi & dugaan penghilangan bukti/liputan": [
            "video", "media", "berita", "hapus", "klarifikasi", "sunarna",
            "tv nasional",
        ],
        "Tuntutan eksplisit keterbukaan pemerintah": [
            "transparan", "jujur", "terbuka",
        ],
    },
}

FALLBACK_LABEL = "Lainnya/tidak terklasifikasi"


def assign_theme(text: str, theme_kw_map: dict) -> str:
    """Mengembalikan nama tema pertama yang cocok dengan kata kunci pada teks."""
    if not isinstance(text, str):
        return FALLBACK_LABEL
    for theme_name, keywords in theme_kw_map.items():
        pattern = re.compile("|".join(keywords))
        if pattern.search(text):
            return theme_name
    return FALLBACK_LABEL


def run_thematic_coding(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan coding tematik pada seluruh komentar bersentimen Negatif."""
    df = df.copy()
    df["tema"] = FALLBACK_LABEL

    negatif_mask = df["predicted_sentiment"] == "Negatif"

    for aspek, theme_kw_map in THEME_DICTIONARY.items():
        aspek_mask = negatif_mask & (df["predicted_aspect"] == aspek)
        df.loc[aspek_mask, "tema"] = df.loc[aspek_mask, "clean_text"].apply(
            lambda t: assign_theme(t, theme_kw_map)
        )

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rekap frekuensi & persentase tema per aspek (hanya komentar Negatif)."""
    neg = df[df["predicted_sentiment"] == "Negatif"].copy()
    summary = (
        neg.groupby(["predicted_aspect", "tema"])
        .size()
        .reset_index(name="jumlah_komentar")
    )
    total_per_aspek = neg.groupby("predicted_aspect").size().rename("total_negatif")
    summary = summary.merge(total_per_aspek, on="predicted_aspect")
    summary["persentase (%)"] = (
        summary["jumlah_komentar"] / summary["total_negatif"] * 100
    ).round(1)
    summary = summary.sort_values(
        ["predicted_aspect", "jumlah_komentar"], ascending=[True, False]
    )
    return summary[
        ["predicted_aspect", "tema", "jumlah_komentar", "total_negatif", "persentase (%)"]
    ]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan di {INPUT_PATH}. "
            "Pastikan predict_absa.py sudah dijalankan lebih dulu."
        )

    df = pd.read_csv(INPUT_PATH)
    coded_df = run_thematic_coding(df)
    summary_df = build_summary(coded_df)

    OUTPUT_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    coded_df.to_csv(OUTPUT_DETAIL_PATH, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print(f"[OK] Detail coding disimpan di: {OUTPUT_DETAIL_PATH}")
    print(f"[OK] Ringkasan tema disimpan di: {OUTPUT_SUMMARY_PATH}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()