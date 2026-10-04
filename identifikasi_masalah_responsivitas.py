"""
identifikasi_masalah_responsivitas.py

Mengidentifikasi INTI MASALAH pada setiap komentar bersentimen Negatif
khusus aspek Responsivitas (65 komentar), berdasarkan pembacaan manual
per komentar. Setiap komentar WAJIB memiliki 1 kategori masalah (termasuk
kategori "Fragmen ekspresi tanpa konten teridentifikasi" untuk komentar
yang genuinely tidak menyebut isu institusional spesifik), sehingga total
persentase = 100% dari 65 komentar.

Input : data/results/indobert_absa_result.csv
Output: data/results/bukti_komentar_responsivitas.csv -> BUKTI per-komentar
        (No, Komentar, Kategori Masalah) -- untuk dokumentasi/pembuktian
        sidang
        data/results/rekap_masalah_responsivitas.csv  -> rekap jumlah &
        persentase tiap kategori masalah (total = 100%)
"""

import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/results/indobert_absa_result.csv")
OUTPUT_DETAIL_PATH = Path("data/results/bukti_komentar_responsivitas.csv")
OUTPUT_SUMMARY_PATH = Path("data/results/rekap_masalah_responsivitas.csv")

ASPEK = "Responsivitas"

# ---------------------------------------------------------------------------
# Kamus kata kunci per kategori masalah.
# Disusun dari HASIL PEMBACAAN MANUAL seluruh 65 komentar negatif aspek
# Responsivitas (bukan asumsi awal), lalu diverifikasi ulang per kategori
# dengan menampilkan seluruh komentar anggotanya untuk mengecek kesesuaian.
# Urutan pengecekan MENENTUKAN prioritas: satu komentar hanya dapat masuk ke
# SATU kategori (kategori pertama yang cocok, sesuai urutan dictionary ini).
# Kategori yang lebih spesifik (tuntutan transparansi, kritik lambatnya
# respon isu lain, sinisme gaya bicara) ditaruh di depan kategori umum
# (pejabat cuma menyindir tanpa tindakan, warga pasif) supaya tidak
# "tertelan" duluan oleh kata kunci umum seperti "ngomong"/"diam" yang
# sangat sering muncul bersamaan.
#
# CATATAN REVISI: ditambahkan kata kunci "baru nyadar"/"baru menyadari"/
# "baru sadar" pada kategori 5, untuk menangkap komentar yang menyoroti
# kesadaran/perhatian pejabat yang baru muncul setelah tekanan publik
# (sebelumnya salah masuk ke kategori fallback karena tidak match kata
# kunci apa pun, padahal isinya substantif).
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "1. Tuntutan transparansi/klarifikasi resmi (tag akun pejabat, minta konten/video/update resmi)": [
        "tag", "akun gubernur", "tiktoknya gubernur", "video yaaa",
        "d kunjungi", "keliling belum",
    ],
    "2. Kritik lambatnya respon pemerintah terhadap isu lain (bencana, tambang liar, dll)": [
        "telat respon", "lambat\\b", "mandek", "adem ayem",
    ],
    "3. Sinisme/ejekan terhadap gaya bicara personal pejabat": [
        "cara ngomong", "pinter ngomong", "pinter omong", "ngeselin",
        "cara ngomongnya", "ngomongnya manis", "susah di omong",
    ],
    "4. Minimnya suara publik Kaltim menanggapi isu ini": [
        "diam", "diem", "bersuara", "protes", "publik kaltim", "warganya kok",
        "demo", "mana ni masyarakat", "mana masyarakat",
    ],
    "5. Pejabat/pemerintah dinilai hanya menyindir/berwacana/bicara tanpa tindakan nyata (termasuk kesadaran yang baru muncul setelah tekanan publik)": [
        "sindir", "nyindir", "tindak", "ngomong", "omong", "omng", "gosip",
        "tegur", "sidak", "bicara", "omon", "peraturan", "pidato", "bilang",
        "baru nyadar", "baru menyadari", "baru sadar",
    ],
    "6. Komentar di luar konteks isu ini (noise/tidak relevan)": [
        "follow", "\\bfb\\b", "fokus ke sini", "tidak di edit",
        "prabowo bilang mobil", "marwah dari kaltim", "tepuk tangan",
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
    # komentar kategori Warga pasif").
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
    print(f"Total komentar negatif Responsivitas: {total}\n")
    print(rekap.to_string(index=False))


if __name__ == "__main__":
    main()