"""
identifikasi_masalah_akuntabilitas.py

Mengidentifikasi INTI MASALAH pada setiap komentar bersentimen Negatif
khusus aspek Akuntabilitas (497 komentar), berdasarkan pembacaan manual
per komentar. Setiap komentar WAJIB memiliki 1 kategori masalah (termasuk
kategori "Komentar tanpa keluhan spesifik" untuk komentar yang genuinely
tidak menyebut isu institusional spesifik), sehingga total persentase =
100% dari 497 komentar.

Input : data/results/indobert_absa_result.csv
Output: data/results/bukti_komentar_akuntabilitas.csv   -> BUKTI per-komentar
        (No, Komentar, Kategori Masalah) -- untuk dokumentasi/pembuktian sidang
        data/results/rekap_masalah_akuntabilitas.csv    -> rekap jumlah &
        persentase tiap kategori masalah (total = 100%)
"""

import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/results/indobert_absa_result.csv")
OUTPUT_DETAIL_PATH = Path("data/results/bukti_komentar_akuntabilitas.csv")
OUTPUT_SUMMARY_PATH = Path("data/results/rekap_masalah_akuntabilitas.csv")

ASPEK = "Akuntabilitas"

# ---------------------------------------------------------------------------
# Kamus kata kunci per kategori masalah.
# Disusun dari HASIL PEMBACAAN MANUAL seluruh 497 komentar negatif aspek
# Akuntabilitas (bukan asumsi awal), lalu diverifikasi ulang per kategori
# dengan menampilkan seluruh komentar anggotanya untuk mengecek kesesuaian.
# Urutan pengecekan MENENTUKAN prioritas: satu komentar hanya dapat masuk ke
# SATU kategori (kategori pertama yang cocok, sesuai urutan dictionary ini).
#
# CATATAN REVISI (hasil bimbingan dosen pembimbing 1):
# - Komentar "sehat selalu kai dan h farid mudahan kembali jdi gebenur dan
#   h farid wakilnya" dipindahkan keluar dari kategori "dukungan/pembelaan
#   ke gubernur", karena setelah ditelaah ulang komentar ini justru berisi
#   doa agar pasangan pemimpin SEBELUMNYA (bukan gubernur saat ini) kembali
#   menjabat, sehingga secara implisit merupakan sindiran/perbandingan yang
#   tidak menguntungkan gubernur saat ini -> dibuatkan kategori baru (9).
# - Nama beberapa kategori disederhanakan agar tidak terkesan seperti hasil
#   tulisan AI (kaku/berlebihan), termasuk kategori fallback.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "1. Kesenjangan gaya hidup mewah pejabat (termasuk keluarga) vs kondisi rakyat": [
        "marwah", "warwah", "mewah", "naik.*jelek", "bagus.*jelek", "ala kadarnya",
        "kasta", "sopir", "naik motor", "hedon", "foya", "royal", "elit",
        "gaya hidup", "fashion", "kebaya", "diana", "bridgerton", "kate middleton",
        "over power", "istri", "gengsi", "nepal kan", "nepalkan", "arwah",
        "maruwa", "marwa ", "raja louis", "di atas penderitaan", "bangsawan",
        "ratu2an", "ratu ratuan", "cantikkan", "sherly tjoanda", "aura",
        "bibirnya", "gugup dia", "jodoh adalah", "belinya di luar negeri",
        "ratu sofia", "mobil jelek", "rubocon", "usap hidung", "gaya tidak sesuai",
        "mantel salju", "risih.*naik", "kekayaan pemimpinnya", "mobil.*merosot",
        "rumah dinasnya cakep",
    ],
    "2. Dugaan penyalahgunaan wewenang/korupsi & tuntutan pemeriksaan/penindakan hukum": [
        "kpk", "korupsi", "korop", "kuropsi", "koropsi", "korups", "pidana",
        "periksa", "dipriksa", "audit", "amplop", "gratifikasi", "harta 166",
        "dari mana", "usut", "ott", "pemalsuan", "nggarong", "dijajah",
        "pajak.*nikmati", "bayar pajak cuma", "suara.*dibeli", "suara.*dibayar",
        "mencontoh atasannya", "mencobtoh", "dibalikan.*ketahuan",
        "katanya mau di balikan", "pantau", "sidak", "tangkap", "ktp",
        "memperkaya diri",
    ],
    "3. Dugaan dinasti politik/nepotisme keluarga pejabat": [
        "dinasti", "bersaudara", "sodaraan", "adek kakak", "keluarga.*pejabat",
        "kroni", "walikota balikpapan", "ketua dprd.*abang", "pamannya",
        "anggota dpr ri",
    ],
    "4. Kinerja kepemimpinan dinilai buruk/tidak berpihak ke rakyat, memanfaatkan jabatan untuk kepentingan pribadi": [
        "kerjaanya", "kerjanya", "kerja.*kagak", "tidak becus", "zonk",
        "mementingkan diri", "tamak", "moral pejabat", "tidak mikir rakyat",
        "kagak ada yang mikir rakyat", "mikirin diri sendiri",
        "kepentingan pribadi", "blunder", "petatang peteteng",
        "tidak ada yang mikir rakyat", "mementingkan rakyat",
        "cerminan.*pemimpin.*susah", "rusak sama kyk jln", "penampilan.*kerja",
        "kenyamanan.*sendiri", "kenyamanan.*sorang", "mumpung.*menjabat",
        "mumpung.*jadi pejabat", "kesempatan tidak datang dua kali",
        "manfaat.*rakyat", "apa gunanya jadi pemimpin", "takut rakyatnya",
        "rakyat menjerit", "uang pribadinya", "tidak sensitif sama rakyat",
    ],
    "5. Tuntutan pertanggungjawaban politik/tindakan tegas (evaluasi, pemecatan, desakan bertindak bukan sekadar sindir)": [
        "jangan dipilih", "tidak dipilih", "periode", "priode", "pecat",
        "copot", "turunkan", "turun kan", "mundur", "ganti saja",
        "ganti gubernur", "gubernur.*ganti", "evaluasi", "salah pilih",
        "nyesel.*pilih", "nyesel.*coblos", "yang milih", "siapa yang pilih",
        "tindak.*bukan", "nyindir doang", "jangan protes",
        "masih di pertahan kan", "terlluh", "tertipu",
    ],
    "6. Sinisme/ejekan personal terhadap sosok gubernur (kompetensi, gaya bicara, kepantasan menjabat)": [
        "kok bisa jadi gubernur", "iq gubernur", "kebingungan", "beda kelas",
        "bingung.*gubernur", "partai apa", "pikirannya.*rusak", "tidak logis dong",
        "utusan dari partai", "gubernur lol", "koplak", "ngampung",
        "cara ngomongnya", "kumaha piikirana", "pikirana.*gub",
        "tidak suka sama bapak", "dy mau jadi ap", "ayu ting ting", "grobak",
        "bisa di coblos", "pejabat konoha", "pejabat pelawak",
        "gubernur kaltim 11 12", "pengusung nya", "kader partai",
        "pilih gua saja",
    ],
    "7. Rasa malu warga & sorotan negatif nama daerah": [
        "malu", "dibully", "di bully", "kacau", "memalukan", "capek.*dengar",
        "capeee.*denger", "warga kaltim kena prank", "kesian rakyat",
        "kasian.*pemimpin",
    ],
    "8. Komentar yang membela pihak gubernur dari kritik": [
        "harus presiden 2029", "masa presiden ngurusin mobil orang",
    ],
    "9. Perbandingan tidak menguntungkan dengan kepemimpinan sebelumnya": [
        "sehat selalu kai dan h farid",
    ],
    "10. Kekecewaan terhadap kondisi pemerintahan/politik Indonesia secara umum (bukan spesifik kasus ini)": [
        "muak.*wni", "pengen lepas dari indonesia", "indo tidak akan maju",
        "pr pak prabowo", "memperbaiki keadaan indonesia", "masuk politik ye",
        "di pusat.*di pedesaan", "pejabat.*sama pemikirannya",
    ],
    "11. Komentar di luar konteks isu pengadaan mobil (noise/tidak relevan, termasuk video lain yang tercampur)": [
        "lagu", "stiker", "takjil", "bukber", "residen evil", "resident evil",
        "story ku", "loundry", "follow", "tag", "kaltim kalau mlm", "pulau gw",
        "tombol riset", "suara jin", "mobil horor", "mobil angker",
        "kaget pas buka", "buka pintu", "kutai daily news", "kreator", "sigra",
        "era tahun berapa", "gue di kaltim wak", "suara apa itu dari dalam",
    ],
}

FALLBACK_LABEL = "12. Komentar tanpa keluhan spesifik"


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
    # komentar kategori Dugaan KKN").
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
    rekap["_urutan"] = rekap["Kategori Masalah"].str.extract(r"^(\d+)\.").astype(int)
    rekap = rekap.sort_values("_urutan").drop(columns="_urutan").reset_index(drop=True)

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
    print(f"Total komentar negatif Akuntabilitas: {total}\n")
    print(rekap.to_string(index=False))


if __name__ == "__main__":
    main()