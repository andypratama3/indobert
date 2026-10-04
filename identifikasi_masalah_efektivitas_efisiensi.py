"""
identifikasi_masalah_efektivitas_efisiensi.py

Mengidentifikasi INTI MASALAH pada setiap komentar bersentimen Negatif
khusus aspek Efektivitas dan Efisiensi (300 komentar), berdasarkan
pembacaan manual per komentar. Setiap komentar WAJIB memiliki 1 kategori
masalah (termasuk kategori "Fragmen ekspresi tanpa konten teridentifikasi"
untuk komentar yang genuinely tidak menyebut isu institusional spesifik),
sehingga total persentase = 100% dari 300 komentar.

Input : data/results/indobert_absa_result.csv
Output: data/results/bukti_komentar_efektivitas_efisiensi.csv -> BUKTI
        per-komentar (No, Komentar, Kategori Masalah) -- untuk dokumentasi/
        pembuktian sidang
        data/results/rekap_masalah_efektivitas_efisiensi.csv  -> rekap
        jumlah & persentase tiap kategori masalah (total = 100%)
"""

import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/results/indobert_absa_result.csv")
OUTPUT_DETAIL_PATH = Path("data/results/bukti_komentar_efektivitas_efisiensi.csv")
OUTPUT_SUMMARY_PATH = Path("data/results/rekap_masalah_efektivitas_efisiensi.csv")

ASPEK = "Efektivitas dan Efisiensi"

# ---------------------------------------------------------------------------
# Kamus kata kunci per kategori masalah.
# Disusun dari HASIL PEMBACAAN MANUAL seluruh 300 komentar negatif aspek
# Efektivitas dan Efisiensi (bukan asumsi awal), lalu diverifikasi ulang per
# kategori dengan menampilkan seluruh komentar anggotanya untuk mengecek
# kesesuaian. Urutan pengecekan MENENTUKAN prioritas: satu komentar hanya
# dapat masuk ke SATU kategori (kategori pertama yang cocok, sesuai urutan
# dictionary ini). Kategori yang lebih spesifik (pemborosan anggaran lain,
# dugaan penyalahgunaan aset) ditaruh di depan kategori umum (jalan vs
# mobil, harga mobil) supaya tidak "tertelan" duluan oleh kata kunci umum
# seperti "mobil"/"anggaran" yang sangat sering muncul bersamaan.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "1. Pemborosan anggaran lain di luar mobil dinas (rumah dinas, toilet renovasi, pesawat/jet pribadi, dll)": [
        "rumah dinas", "toilet", "pesawat pribadi", "jet pribadi", "jed pribadi",
        "air jet", "renovasi rumah", "25 ?m\\b", "300 miliyar", "300 milyar",
    ],
    "2. Dugaan penyalahgunaan status kepemilikan aset dinas pasca masa jabatan": [
        "di dum", "jadi milik pribadi", "gratis kalo", "lsng untuk dia",
        "kalo tidak menjabat",
    ],
    "3. Prioritas anggaran dinilai keliru: jalan/infrastruktur diabaikan demi mobil dinas mewah": [
        "jalan", "jln\\b", "jalana", "infrastruktur", "aspal", "hotmix",
        "berlubang", "off ?road", "ofrod", "lumpur", "becek", "jembatan",
        "medan", "tanah liat", "setapak", "sarana", "prasarana", "gerbang",
    ],
    "4. Harga/nilai mobil dinas dinilai kemahalan/boros, termasuk saran merek alternatif lebih murah": [
        "mobil", "mbil\\b", "kendaraan", "8 ?5 ?m", "8m\\b", "700 ?jt",
        "700 juta", "700jt",
        "pajero", "hilux", "hiluk", "hilluk", "triton", "fortuner",
        "avanza", "avzra", "innova", "inova", "zenix", "alphard", "alpard",
        "range rover", "land cruiser", "raptor", "panther", "sigra",
        "maung", "mux", "isuzu", "ford", "toyota", "lexus", "grenmax",
        "sedan", "jeep", "kijang", "angkot", "roket", "roll royce",
        "motor", "sepeda", "honda beat", "mio", "stnk", "gengsi",
        "efisiensi", "mewah", "marwah", "truk",
    ],
    "5. Kritik penggunaan anggaran/uang pajak rakyat untuk hal yang tidak esensial": [
        "pajak", "uang rakyat", "duit rakyat", "ngabisin duit",
        "kepentingannya sendiri", "anggaran di pake",
    ],
    "6. Komentar di luar konteks isu pengadaan mobil (noise/tidak relevan)": [
        "tombol reset", "wafer nisin", "magrib", "stiker", "es nya",
        "kipas nya", "kelurahan pang", "sim ya", "pppk", "gaji", "200 ribu",
        "lahan prabowo", "grobak",
    ],
}

FALLBACK_LABEL = "7. Komentar tanpa keluhan spesifik"


def assign_masalah(text: str) -> str:
    """Menentukan kategori masalah berdasarkan kata kunci hasil coding manual."""
    if not isinstance(text, str):
        return FALLBACK_LABEL
    for category, keywords in CATEGORY_KEYWORDS.items():
        pattern = re.compile("|".join(keywords))
        if pattern.search(text):
            return category
    return FALLBACK_LABEL


def largest_remainder_round(values: pd.Series, decimals: int = 1) -> pd.Series:
    """Membulatkan sekumpulan persentase ke n desimal dengan largest remainder
    method, sehingga totalnya tetap tepat 100.0 (mengatasi selisih akibat
    pembulatan independen tiap baris)."""
    factor = 10 ** decimals
    scaled = values * factor
    floored = scaled.apply(lambda x: int(x // 1))
    remainder = scaled - floored
    deficit = int(round(100 * factor)) - int(floored.sum())
    order = remainder.sort_values(ascending=False).index
    result = floored.copy()
    for idx in order[:deficit]:
        result[idx] += 1
    return result / factor


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File input tidak ditemukan di {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    akun = df[
        (df["predicted_sentiment"] == "Negatif") & (df["predicted_aspect"] == ASPEK)
    ].copy()
    akun = akun.reset_index(drop=True)

    akun["masalah"] = akun["clean_text"].apply(assign_masalah)

    # Urutkan berdasarkan kategori masalah (angka di depan nama kategori),
    # supaya semua komentar dengan kategori yang sama berkelompok bersebelahan
    # -> memudahkan penelusuran bukti saat sidang (mis. "tunjukkan semua
    # komentar kategori Harga mobil kemahalan").
    akun["_urutan_kategori"] = akun["masalah"].str.extract(r"^(\d+)\.").astype(int)
    akun = akun.sort_values(
        ["_urutan_kategori", "clean_text"]
    ).drop(columns="_urutan_kategori").reset_index(drop=True)

    # --- File bukti per-komentar (senjata untuk sidang), sudah terurut per
    # kategori dan dilengkapi metadata sumber asli (username, URL komentar,
    # URL postingan, hasil prediksi model) agar setiap baris dapat ditelusuri
    # kembali ke sumber TikTok aslinya. ---
    bukti = pd.DataFrame({
        "No": range(1, len(akun) + 1),
        "Username Komentator": akun["comment_username"],
        "Komentar": akun["clean_text"],
        "URL Profil Komentator": akun["comment_profile_url"],
        "URL Postingan TikTok": akun["post_url"],
        "Aspek Prediksi": akun["predicted_aspect"],
        "Sentimen Prediksi": akun["predicted_sentiment"],
        "Kategori Masalah": akun["masalah"],
    })

    # --- Rekap jumlah & persentase per kategori (total harus tepat 100.0%) ---
    total = len(akun)
    rekap = (
        akun["masalah"]
        .value_counts()
        .rename_axis("Kategori Masalah")
        .reset_index(name="Jumlah Komentar")
    )
    persen_asli = rekap["Jumlah Komentar"] / total * 100
    rekap["Persentase (%)"] = largest_remainder_round(persen_asli, decimals=1)
    rekap = rekap.sort_values("Kategori Masalah").reset_index(drop=True)

    # baris Total, untuk verifikasi visual di file rekap
    total_row = pd.DataFrame([{
        "Kategori Masalah": "TOTAL",
        "Jumlah Komentar": rekap["Jumlah Komentar"].sum(),
        "Persentase (%)": round(rekap["Persentase (%)"].sum(), 1),
    }])
    rekap = pd.concat([rekap, total_row], ignore_index=True)

    OUTPUT_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bukti.to_csv(OUTPUT_DETAIL_PATH, index=False)
    rekap.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print(f"[OK] Bukti per-komentar disimpan di : {OUTPUT_DETAIL_PATH}")
    print(f"[OK] Rekap kategori disimpan di     : {OUTPUT_SUMMARY_PATH}\n")
    print(f"Total komentar negatif Efektivitas dan Efisiensi: {total}\n")
    print(rekap.to_string(index=False))


if __name__ == "__main__":
    main()