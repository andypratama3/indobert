"""
TikTok comment scraper untuk korpus skripsi ABSA.

Kenapa file ini ada (BUG-09)
---------------------------
`data/raw/tiktok_comments_mobil_dinas_kaltim.csv` (1081 baris, 7 kolom) adalah
input paling penting di skripsi ini, tapi scraper yang ada di repo
(`scraper/scraper.py`) hanya bisa dipakai untuk Twitter/X. Tidak ada scraper
TikTok yang ter-commit, sehingga data raw tidak bisa di-regenerate oleh siapa
pun yang meng-clone repo ini. File ini menutup celah tersebut.

Skema output (WAJIB persis, urutan kolom tidak boleh diubah)
-----------------------------------------------------------
    keyword, post_url, post_username, comment_username,
    comment_profile_url, comment_text, comment_like_count

Catatan penting soal `comment_like_count`
----------------------------------------
Nilainya disimpan sebagai TEKS apa adanya seperti yang tampil di UI TikTok
("1461", "15K", "26K"), bukan integer hasil parse. CSV yang sudah ter-commit
juga begitu: dtype object, memuat "15K" dan "26K", plus 14 baris kosong. Kalau
kolom ini di-parse jadi integer, hasil regenerate tidak akan identik dengan
file asli yang dipakai di skripsi.

Catatan soal `keyword`
----------------------
Kolom `keyword` berisi query yang menemukan video tersebut, BUKAN teks
komentar. File yang ter-commit memakai 20 keyword spesifik (lihat
`TIKTOK_KEYWORDS`), sementara default skrip ini adalah 32 query yang sama
dengan `run.py` supaya konsisten dengan scraper Twitter. Lewati `--keywords`
kalau mau mereproduksi 20 keyword yang menghasilkan file asli.

Login
-----
Komentar video TikTok bersifat publik dan bisa dibaca tanpa login, jadi tidak
ada login yang WAJIB dilakukan. `login_manual()` secara default hanya memanaskan
sesi lalu langsung lanjut, beda dengan scraper Twitter yang mewajibkan login
karena balasan di X butuh akun. Kalau TikTok mulai menampilkan login-wall,
isi cookie secara manual lewat argumen `--cookie` atau panggil
`scraper.login_manual()` secara eksplisit.

Rate limit dan etika scraping
-----------------------------
- Setiap load halaman dan setiap putaran scroll diberi jeda acak (polite delay),
  bukan request beruntun.
- Kalau halaman terdeteksi rate-limited atau memunculkan CAPTCHA, scraper
  BERHENTI, masuk cooldown, dan tidak memaksa. Memaksa hanya memperburuk
  blokir IP.
- Auto-save tiap selesai satu video, jadi program bisa di-restart dan lanjut
  dari/baris yang sudah ada (resumable).
- Duplicate-safe lewat `drop_duplicates` pada (comment_username, comment_text).
- Scraping ini untuk riset akademik atas data publik saja.

Cara pakai
----------
    python -m scraper.tiktok_scraper                     # 32 keyword default
    python -m scraper.tiktok_scraper --dry-run           # cek rencana, tanpa jaringan
    python -m scraper.tiktok_scraper --keywords "mobil dinas kaltim"
    python -m scraper.tiktok_scraper --target 2000 --resume
"""

import argparse
import os
import random
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# 32 query yang sama dengan run.py supaya daftar keyword repo ini konsisten.
TIKTOK_KEYWORDS = [
    # Query original
    "mobil dinas kaltim",
    "mobil dinas gubernur kaltim",
    "8,5 miliar kaltim",
    "8.5 miliar gubernur kaltim",
    "mobil mewah kaltim",
    "rudy masud",
    "rudy masud kaltim",
    "gubernur kaltim",
    "rumah dinas kaltim",
    "renovasi rumah dinas kaltim",
    "25 miliar kaltim",
    "kursi pijat kaltim",
    "laundry kaltim",
    "marwah kaltim",
    "jaga marwah kaltim",
    "efisiensi kaltim",
    "anggaran kaltim",

    # Opsi 2 - query baru variasi
    "kaltim gubernur",
    "rudy mas ud",
    "pemprov kaltim boros",
    "kaltim viral",
    "gubernur kaltim mewah",
    "mobil mewah pejabat kaltim",
    "pemprov kaltim",
    "kaltim anggaran mewah",
    "rudy masud viral",
    "gubernur kaltim kritik",
    "kaltim 8 miliar",
    "istri gubernur kaltim",
    "dinasti politik kaltim",
    "nepotisme kaltim",
    "korupsi kaltim",
]

# 20 keyword yang benar-benar dipakai untuk menghasilkan CSV yang ter-commit,
# diekstrak langsung dari kolom `keyword` di file itu. Dipakai kalau butuh
# reproduksi file asli persis (`--raw-keywords`).
TIKTOK_KEYWORDS_RAW = [
    "mobil dinas gubernur kaltim",
    "mobil dinas kaltim",
    "pengadaan mobil dinas gubernur kaltim",
    "isu mobil dinas gubernur kaltim",
    "pengadaan mobil dinas gubernur kaltim 8,5 miliar",
    "mobil dinas kaltim marwah",
    "mobil dinas gubernur kaltim 8 5 miliar",
    "pengadaan mobil dinas gub kaltim 8 5 miliar",
    "mobil dinas kaltim 8 5 miliar",
    "mobil dinas baru gubernur kaltim",
    "mobil dinas gub kaltim 8,5 miliar",
    "mobil dinas gubernur kaltim 8,5 miliar",
    "mobil dinas gub kaltim 8 5 miliar",
    "isu pengadaan mobil dinas kaltim",
    "mobil dinas gub kaltim",
    "pengadaan mobil dinas gub kaltim",
    "pengadaan mobil dinas gub kaltim 8,5 miliar",
    "pengadaan mobil dinas gubernur kaltim 8 5 miliar",
    "pengadaan mobil dinas kaltim",
    "mobil dinas gubernur kaltim marwah",
]

# Skema CSV final. Urutan ini kontrak dengan data/ dan skripsi - jangan diubah.
OUTPUT_COLUMNS = [
    "keyword",
    "post_url",
    "post_username",
    "comment_username",
    "comment_profile_url",
    "comment_text",
    "comment_like_count",
]

DEDUPE_SUBSET = ["comment_username", "comment_text"]

OUTPUT_DIR = "data/raw"
OUTPUT_FILENAME = "tiktok_comments_mobil_dinas_kaltim.csv"

TIKTOK_HOME = "https://www.tiktok.com/"
TIKTOK_SEARCH = "https://www.tiktok.com/search?q={query}"
TIKTOK_PROFILE_PREFIX = "https://www.tiktok.com/"

# Kata penanda rate limit / CAPTCHA. Kalau muncul, jeda, jangan dipaksa.
RATE_LIMIT_MARKERS = [
    "too many requests",
    "rate limit",
    "access denied",
    "verify to continue",
    "captcha",
    "unusual traffic",
    "please try again later",
]


class TikTokScraper:
    """
    Scraper komentar TikTok berbasis Selenium.

    Mengikuti konvensi `scraper/scraper.py` (kelas, setup/login/search/scrape,
    auto-save ke CSV, dan `data_comments` sebagai penampung baris), tapi
   tapi untuk TikTok dan tidak butuh login.
    """

    def __init__(
        self,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        delay_range=(3.0, 6.0),
        polite_delay_range=(8.0, 15.0),
        require_login=False,
        cookie=None,
    ):
        self.seed_posts = []
        self.data_comments = []
        self.used_seed_urls = set()
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.delay_range = delay_range
        self.polite_delay_range = polite_delay_range
        self.require_login = require_login
        self.cookie = cookie
        self.driver = None
        self.consecutive_failures = 0

    # ------------------------------------------------------------------
    # Utilitas kecil
    # ------------------------------------------------------------------
    @property
    def output_path(self):
        return os.path.join(self.output_dir, self.output_filename)

    def clean_text(self, text):
        return " ".join(str(text).split())

    def polite_delay(self):
        """Jeda acak supaya tidak membanjiri TikTok dengan request."""
        time.sleep(random.uniform(*self.polite_delay_range))

    def scroll_delay(self):
        """Jeda lebih pendek, dipakai antar putaran scroll."""
        time.sleep(random.uniform(*self.delay_range))

    def normalize_url(self, url):
        if not url:
            return ""

        url = str(url).strip()
        url = url.split("?")[0].split("#")[0]
        if not url.startswith("http"):
            url = TIKTOK_PROFILE_PREFIX + url.lstrip("/")

        # Buang trailing slash supaya "https://www.tiktok.com/@user" dan
        # "https://www.tiktok.com/@user/" dianggap sama.
        return url.rstrip("/")

    def extract_username_from_url(self, url):
        """Ambil '@username' dari URL profil atau video TikTok."""
        match = re.search(r"tiktok\.com/@([^/?#]+)", str(url or ""))
        if not match:
            return ""

        return "@" + match.group(1)

    def profile_url(self, username):
        if not username:
            return ""

        return TIKTOK_PROFILE_PREFIX + "@" + str(username).lstrip("@")

    def parse_count(self, value):
        """
        Ubah teks count TikTok seperti: 12 -> 12, 1.2K -> 1200, 15K -> 15000.

        Hanya dipakai untuk membandingkan/memfilter. Kolom
        `comment_like_count` di CSV tetap ditulis sebagai teks mentah supaya
        konsisten dengan file raw yang sudah ada.
        """
        if value is None:
            return 0

        text = str(value).strip().upper().replace(",", "")
        if not text:
            return 0

        try:
            if text.endswith("K"):
                return int(float(text[:-1]) * 1000)
            if text.endswith("M"):
                return int(float(text[:-1]) * 1000000)
            if text.endswith("B"):
                return int(float(text[:-1]) * 1000000000)
            return int(float(text))
        except Exception:
            return 0

    # Term list di bawah sengaja identik dengan TwitterScraper supaya filter
    # seed konsisten di kedua scraper.
    def topic_score(self, text):
        text = self.clean_text(text).lower()

        strong_phrases = [
            "mobil gubernur kaltim",
            "mobil dinas gubernur kaltim",
            "mobil dinas gub kaltim",
            "mobil dinas 8,5 miliar",
            "mobil dinas 8.5 miliar",
            "mobil dinas 8,5m",
            "mobil dinas 8.5m",
            "gubernur kaltim 8,5 m",
            "gubernur kaltim 8.5 m",
            "mobil dinas kaltim",
            "mobil gubernur",
        ]

        primary_terms = [
            "mobil",
            "dinas",
            "gubernur",
            "gub",
            "kaltim",
            "8,5",
            "8.5",
            "miliar",
            "marwah",
        ]

        noise_terms = [
            "ikn",
            "pilkada",
            "partai",
            "wagub",
            "wakil gubernur",
            "banjir",
            "jalan",
            "jembatan",
            "rumah jabatan",
        ]

        score = 0

        for phrase in strong_phrases:
            if phrase in text:
                score += 8

        for term in primary_terms:
            if term in text:
                score += 2

        for term in noise_terms:
            if term in text:
                score -= 1

        return score

    def is_relevant(self, text):
        text = self.clean_text(text).lower()

        strong_phrases = [
            "mobil gubernur kaltim",
            "mobil dinas gubernur kaltim",
            "mobil dinas gub kaltim",
            "mobil dinas 8,5 miliar",
            "mobil dinas 8.5 miliar",
            "mobil dinas 8,5m",
            "mobil dinas 8.5m",
            "mobil dinas kaltim",
        ]
        if any(phrase in text for phrase in strong_phrases):
            return True

        has_vehicle = "mobil" in text
        has_official = ("dinas" in text) or ("gubernur" in text) or ("gub" in text)
        has_region = "kaltim" in text
        has_money = any(k in text for k in ["8,5", "8.5", "miliar", "8,5m", "8.5m"])

        return has_vehicle and has_official and (has_region or has_money)

    # ------------------------------------------------------------------
    # Driver dan sesi
    # ------------------------------------------------------------------
    def setup_driver(self, headless=False):
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--lang=id-ID")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )

        if headless:
            options.add_argument("--headless=new")

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(60)

        try:
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
            )
        except Exception:
            pass

        return self.driver

    def login_manual(self, timeout=180):
        """
        TikTok tidak butuh login untuk membaca komentar publik.

        Default-nya method ini cuma memanaskan sesi (buka beranda, biarkan
        cookie TikTok terbentuk) lalu langsung lanjut. Kalau `require_login`
        di-set True, barulahmethod ini menunggu login manual seperti
        `TwitterScraper.login_manual()`.
        """
        print("Menyiapkan sesi TikTok (komentar publik tidak butuh login)...")
        self.driver.get(TIKTOK_HOME)
        time.sleep(5)

        if self.cookie:
            for item in self.cookie.split(";"):
                name, _, value = item.strip().partition("=")
                if name and value:
                    self.driver.add_cookie({"name": name, "value": value, "domain": ".tiktok.com"})
            self.driver.refresh()
            time.sleep(3)

        if not self.require_login:
            print("Sesi siap. Lanjut scraping tanpa login.")
            return True

        print("Login manual diminta (require_login=True).")
        print(f"Kamu punya waktu {timeout} detik untuk login di jendela browser.")

        start = time.time()
        while time.time() - start < timeout:
            if "login" not in (self.driver.current_url or "").lower():
                print("Login terdeteksi selesai.")
                time.sleep(5)
                return True

            time.sleep(2)

        print("Login belum selesai sampai timeout.")
        return False

    def close(self):
        try:
            if self.driver is not None:
                self.driver.quit()
        except Exception:
            pass
        finally:
            self.driver = None

    # ------------------------------------------------------------------
    # Deteksi rate limit
    # ------------------------------------------------------------------
    def is_rate_limited(self):
        try:
            body = (self.driver.find_element(By.TAG_NAME, "body").text or "").lower()
        except Exception:
            return False

        if any(marker in body for marker in RATE_LIMIT_MARKERS):
            return True

        try:
            if self.driver.find_elements(By.CSS_SELECTOR, 'iframe[src*="captcha"]'):
                return True
        except Exception:
            pass

        return False

    def handle_rate_limit(self, cooldown=180):
        """Berhenti sejenak kalau kena rate limit. Jangan dipaksa."""
        self.consecutive_failures += 1

        if self.consecutive_failures < 2:
            time.sleep(30)
            return False

        print(f"  Rate limit / blokir terdeteksi. Cooldown {cooldown} detik...")
        self.close()
        time.sleep(cooldown)
        self.setup_driver()
        self.consecutive_failures = 0
        return True

    # ------------------------------------------------------------------
    # Pencarian post (seed)
    # ------------------------------------------------------------------
    def search_posts(self, query, max_posts=40, scroll_loops=20):
        """Cari video lewat halaman search TikTok, lalu scroll hasilnya."""
        url = TIKTOK_SEARCH.format(query=query)
        self.driver.get(url)
        self.polite_delay()

        posts = []
        seen_urls = set()
        no_new_rounds = 0

        for _ in range(scroll_loops):
            before_count = len(seen_urls)

            anchors = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/video/"]')

            for anchor in anchors:
                try:
                    href = self.normalize_url(anchor.get_attribute("href"))
                except StaleElementReferenceException:
                    continue

                if not href or "/video/" not in href or href in seen_urls:
                    continue

                seen_urls.add(href)

                item = {
                    "post_url": href,
                    "post_username": self.extract_username_from_url(href),
                    "query_used": query,
                }
                posts.append(item)

                if len(posts) >= max_posts:
                    break

            if len(posts) >= max_posts:
                break

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.scroll_delay()

            if len(seen_urls) == before_count:
                no_new_rounds += 1
            else:
                no_new_rounds = 0

            if no_new_rounds >= 3:
                break

        # Buang dulu yang URL-nya tidak bisa dibaca username-nya.
        posts = sorted(
            posts,
            key=lambda x: (bool(x["post_username"]), x["post_url"]),
            reverse=False,
        )

        return posts[:max_posts]

    def open_comment_panel(self, post_url, timeout=20):
        """
        Buka panel komentar satu video.

        Dua cara: klik ikon komentar, atau klik count komentar (yang punya href
        langsung ke thread komentar). Kalau dua-duanya gagal, scraper tetap
        mencoba karena TikTok versi terbaru kadang memuat komentar langsung di
        halaman video tanpa panel terpisah.
        """
        self.driver.get(post_url)
        self.polite_delay()

        if self.is_rate_limited():
            return False

        selectors = [
            'a[href*="/video/"][href*="comment"]',
            '[data-e2e="comment-icon"]',
            '[data-e2e="comment-count"]',
            'div[data-e2e="comment-icon"] span',
        ]

        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                continue

            for element in elements:
                try:
                    href = element.get_attribute("href")
                    if href and "comment" in href:
                        self.driver.get(self.normalize_url(href))
                        self.polite_delay()
                        return True

                    self.driver.execute_script("arguments[0].click();", element)
                    self.scroll_delay()
                    if self.find_comment_items():
                        return True
                except (ElementClickInterceptedException, StaleElementReferenceException, WebDriverException):
                    try:
                        element.send_keys(Keys.ESCAPE)
                    except Exception:
                        pass

        return bool(self.find_comment_items())

    def find_comment_items(self):
        """Cari container item komentar dengan beberapa selector fallback."""
        selectors = [
            '[data-e2e="comment-item"]',
            'div[data-e2e="comment-level-1"]',
            "div.DivCommentContentContainer",
            "div[class*='DivCommentContentContainer']",
            'li[data-e2e="comment-item"]',
        ]

        for selector in selectors:
            try:
                items = self.driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                continue

            if items:
                return items

        return []

    # ------------------------------------------------------------------
    # Ekstraksi satu baris komentar
    # ------------------------------------------------------------------
    def extract_comment_username(self, item):
        selectors = [
            '[data-e2e="comment-username-1"]',
            'a[href^="/@"]',
            'a[href*="tiktok.com/@"]',
            "a[href*='/@']",
        ]

        for selector in selectors:
            try:
                for element in item.find_elements(By.CSS_SELECTOR, selector):
                    handle = self.extract_username_from_url(element.get_attribute("href"))
                    if handle:
                        return handle

                    text = self.clean_text(element.text)
                    if text.startswith("@"):
                        return text.split()[0]
            except (StaleElementReferenceException, NoSuchElementException):
                continue

        return ""

    def extract_comment_text(self, item):
        selectors = [
            '[data-e2e="comment-level-1"]',
            'span[data-e2e*="comment-text"]',
            "span[class*='CommentText']",
            "div[class*='CommentText']",
            "p",
        ]

        for selector in selectors:
            try:
                for element in item.find_elements(By.CSS_SELECTOR, selector):
                    text = self.clean_text(element.text)
                    if text:
                        return text
            except (StaleElementReferenceException, NoSuchElementException):
                continue

        return ""

    def extract_comment_like_count(self, item):
        """
        Ambil like count komentar apa adanya ("1461", "15K").

        Sengaja TIDAK di-parse jadi integer supaya kolom CSV sama dengan file
        raw yang sudah ter-commit.
        """
        selectors = [
            '[data-e2e="comment-like-count-1"]',
            'span[data-e2e*="like-count"]',
            "span[class*='LikeCount']",
            "strong",
        ]

        for selector in selectors:
            try:
                for element in item.find_elements(By.CSS_SELECTOR, selector):
                    text = self.clean_text(element.text)
                    if text and self.parse_count(text) >= 0:
                        return text
            except (StaleElementReferenceException, NoSuchElementException):
                continue

        # Tanpa like count -> kosong, sama seperti 14 baris kosong di file asli.
        return ""

    def build_comment_row(self, item, keyword, post_url, post_username):
        """Rakit satu baris sesuai OUTPUT_COLUMNS, atau None kalau tidak valid."""
        comment_text = self.extract_comment_text(item)
        if not comment_text:
            return None

        comment_username = self.extract_comment_username(item)
        if not comment_username:
            return None

        return {
            "keyword": keyword,
            "post_url": post_url,
            "post_username": post_username or self.extract_username_from_url(post_url),
            "comment_username": comment_username,
            "comment_profile_url": self.profile_url(comment_username),
            "comment_text": comment_text,
            "comment_like_count": self.extract_comment_like_count(item),
        }

    # ------------------------------------------------------------------
    # Scroll panel komentar
    # ------------------------------------------------------------------
    def scroll_comment_panel(self):
        """
        TikTok memuat komentar lewat scroll pada container sendiri, bukan
        window. Coba container yang paling tinggi dulu, baru fallback ke window.
        """
        scrolled = self.driver.execute_script(
            """
            var best = null;
            var bestGain = 0;
            var nodes = document.querySelectorAll('div, ul, section');
            for (var i = 0; i < nodes.length; i++) {
                var n = nodes[i];
                var gain = n.scrollHeight - n.clientHeight;
                if (gain > bestGain && n.clientHeight > 200) {
                    bestGain = gain;
                    best = n;
                }
            }
            if (best) {
                best.scrollTop = best.scrollHeight;
                return bestGain;
            }
            window.scrollTo(0, document.body.scrollHeight);
            return 0;
            """
        )

        return bool(scrolled and scrolled > 0)

    # ------------------------------------------------------------------
    # Scrape komentar satu video
    # ------------------------------------------------------------------
    def scrape_comments(self, post_url, keyword="", post_username="", max_scroll=120):
        """Kumpulkan semua komentar yang bisa di-load dari satu video."""
        comments = []
        seen_items = set()
        no_new_rounds = 0
        last_total = 0

        self.open_comment_panel(post_url)

        for _ in range(max_scroll):
            items = self.find_comment_items()

            for item in items:
                try:
                    row = self.build_comment_row(item, keyword, post_url, post_username)
                except StaleElementReferenceException:
                    continue

                if not row:
                    continue

                item_key = f"{row['comment_username']}|{row['comment_text']}"
                if item_key in seen_items:
                    continue

                seen_items.add(item_key)
                comments.append(row)

            self.scroll_comment_panel()
            self.scroll_delay()

            if self.is_rate_limited():
                print("  Rate limit saat scroll komentar, berhenti untuk video ini.")
                break

            current_total = len(comments)
            if current_total == last_total:
                no_new_rounds += 1
            else:
                no_new_rounds = 0

            last_total = current_total

            if no_new_rounds >= 4:
                print("  Scroll mentok, tidak ada komentar baru.")
                break

        return comments

    # ------------------------------------------------------------------
    # Simpan / muat CSV (auto-save + resume)
    # ------------------------------------------------------------------
    def load_existing(self):
        """Baca CSV yang sudah ada. Dipakai untuk resume."""
        if not os.path.exists(self.output_path):
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        try:
            df = pd.read_csv(self.output_path, dtype=str).fillna("")
        except Exception as exc:
            print(f"Gagal baca {self.output_path}: {exc}")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        for column in OUTPUT_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        return df[OUTPUT_COLUMNS]

    def load_existing_keys(self):
        """Set (comment_username, comment_text) yang sudah pernah terkumpul."""
        df = self.load_existing()
        if df.empty:
            return set()

        return set(
            zip(df["comment_username"].astype(str), df["comment_text"].astype(str))
        )

    def load_existing_seed_urls(self):
        """URL video yang sudah pernah diproses, untuk resume."""
        df = self.load_existing()
        if df.empty:
            return set()

        return set(self.normalize_url(u) for u in df["post_url"].astype(str) if str(u).strip())

    def total_rows_saved(self):
        df = self.load_existing()
        return 0 if df.empty else len(df)

    def save_csv(self, df, filename=None):
        filename = filename or self.output_filename
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"File disimpan ke: {filepath}")
        return filepath

    def auto_save(self):
        """
        Gabungkan baris baru dengan yang sudah ada, drop duplikat, tulis ulang.

        Dipanggil setelah setiap selesai satu video supaya program aman di-kill
        dan bisa di-restart (resumable).
        """
        if not self.data_comments:
            return 0

        df_new = pd.DataFrame(self.data_comments)
        for column in OUTPUT_COLUMNS:
            if column not in df_new.columns:
                df_new[column] = ""

        df_new = df_new[OUTPUT_COLUMNS].astype(str).replace({"": None}).fillna("")

        df_old = self.load_existing()
        if not df_old.empty:
            df_old = df_old[OUTPUT_COLUMNS].astype(str).fillna("")
            df_new = pd.concat([df_old, df_new], ignore_index=True)

        df_new = df_new.drop_duplicates(subset=DEDUPE_SUBSET).reset_index(drop=True)
        self.save_csv(df_new)

        print(f"  [AUTO-SAVE] {len(df_new)} baris tersimpan.")
        return len(df_new)

    # ------------------------------------------------------------------
    # Orkestrasi
    # ------------------------------------------------------------------
    def run(
        self,
        queries=None,
        max_posts_per_query=30,
        max_scroll_search=20,
        max_scroll_per_post=120,
        target_rows=1081,
        min_relevant_seeds=0,
        require_relevance=False,
        resume=True,
    ):
        queries = list(queries or TIKTOK_KEYWORDS)

        self.setup_driver()
        login_ok = self.login_manual()
        if not login_ok:
            print("Sesi tidak siap. Program dihentikan.")
            self.close()
            return

        already_keys = self.load_existing_keys() if resume else set()
        already_seeds = self.load_existing_seed_urls() if resume else set()

        print(f"Komentar sudah ada: {len(already_keys)}")
        print(f"Video sudah diproses: {len(already_seeds)}")
        print(f"Target baris: {target_rows}")

        for q_idx, query in enumerate(queries, start=1):
            if self.total_rows_saved() >= target_rows:
                print(f"\nTARGET {target_rows} TERCAPAI. Berhenti.")
                break

            print("\n" + "=" * 80)
            print(f"QUERY [{q_idx}/{len(queries)}]")
            print(query)
            print("=" * 80)

            if self.is_rate_limited() and self.handle_rate_limit():
                continue

            seed_posts = self.search_posts(
                query=query,
                max_posts=max_posts_per_query,
                scroll_loops=max_scroll_search,
            )

            print(f"Total video ditemukan: {len(seed_posts)}")

            relevant_seeds = [
                p
                for p in seed_posts
                if p["post_url"] not in already_seeds
                and p["post_url"] not in self.used_seed_urls
            ]

            if require_relevance:
                strict = [p for p in relevant_seeds if self.is_relevant(p["post_url"])]
                if len(strict) >= min_relevant_seeds:
                    relevant_seeds = strict

            print(f"Video baru yang akan diproses: {len(relevant_seeds)}")
            self.seed_posts.extend(relevant_seeds)

            for i, item in enumerate(relevant_seeds, start=1):
                post_url = item["post_url"]
                self.used_seed_urls.add(post_url)

                print(f"[Q{q_idx} - {i}/{len(relevant_seeds)}] {post_url}")

                try:
                    comments = self.scrape_comments(
                        post_url=post_url,
                        keyword=query,
                        post_username=item["post_username"],
                        max_scroll=max_scroll_per_post,
                    )
                except (WebDriverException, TimeoutException) as exc:
                    print(f"  Gagal scrape: {exc}")
                    if self.handle_rate_limit():
                        self.setup_driver()
                        self.login_manual()
                    continue

                new_rows = [
                    c
                    for c in comments
                    if (c["comment_username"], c["comment_text"]) not in already_keys
                ]

                print(f"  Komentar dari video ini: {len(comments)} (baru: {len(new_rows)})")

                self.data_comments.extend(new_rows)
                for row in new_rows:
                    already_keys.add((row["comment_username"], row["comment_text"]))

                # Auto-save tiap selesai satu video -> resumable.
                total = self.auto_save()
                self.data_comments = []
                self.seed_posts = []

                already_seeds.add(post_url)

                if total >= target_rows:
                    print(f"\nTARGET {target_rows} TERCAPAI. Berhenti.")
                    self.close()
                    print("Scraping selesai.")
                    return

            self.polite_delay()

        self.close()
        print(f"Scraping selesai. Total baris: {self.total_rows_saved()}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_keywords(values):
    if not values:
        return TIKTOK_KEYWORDS

    keywords = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                keywords.append(part)

    return keywords or TIKTOK_KEYWORDS


def main():
    parser = argparse.ArgumentParser(
        description="Scraper komentar TikTok untuk korpus ABSA mobil dinas Kaltim.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Keyword pencarian (pisahkan dengan spasi atau koma). Default: 32 query run.py.",
    )
    parser.add_argument(
        "--raw-keywords",
        action="store_true",
        help="Pakai 20 keyword yang menghasilkan CSV yang ter-commit.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Direktori output (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILENAME,
        help=f"Nama file CSV (default: {OUTPUT_FILENAME}).",
    )
    parser.add_argument("--target", type=int, default=1081, help="Target jumlah baris.")
    parser.add_argument("--max-posts", type=int, default=30, help="Maks video per keyword.")
    parser.add_argument("--max-scroll-post", type=int, default=120, help="Maks putaran scroll per video.")
    parser.add_argument("--max-scroll-search", type=int, default=20, help="Maks putaran scroll search.")
    parser.add_argument("--no-resume", action="store_true", help="Abaikan baris yang sudah ada.")
    parser.add_argument("--require-login", action="store_true", help="Paksa login manual.")
    parser.add_argument("--cookie", default=None, help='Cookie TikTok, mis. "sessionid=...; ttwid=...".')
    parser.add_argument("--headless", action="store_true", help="Jalankan Chrome headless.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cetak rencana lalu keluar. Tidak membuka browser, tidak menyentuh jaringan.",
    )

    args = parser.parse_args()

    if args.raw_keywords:
        keywords = list(TIKTOK_KEYWORDS_RAW)
    else:
        keywords = parse_keywords(args.keywords)

    output_path = os.path.join(args.output_dir, args.output_file)

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN - tidak ada browser dibuka, tidak ada request ke TikTok.")
        print("=" * 80)
        print(f"Jumlah keyword : {len(keywords)}")
        for i, keyword in enumerate(keywords, start=1):
            print(f"  [{i:2d}] {keyword}")
        print(f"Target baris   : {args.target}")
        print(f"Output         : {output_path}")
        print(f"Skema kolom    : {', '.join(OUTPUT_COLUMNS)}")
        print(f"Dedupe subset  : {', '.join(DEDUPE_SUBSET)}")
        return 0

    scraper = TikTokScraper(
        output_dir=args.output_dir,
        output_filename=args.output_file,
        require_login=args.require_login,
        cookie=args.cookie,
    )

    if args.headless:
        scraper.setup_driver(headless=True)
        scraper.login_manual()
    else:
        scraper.run(
            queries=keywords,
            max_posts_per_query=args.max_posts,
            max_scroll_search=args.max_scroll_search,
            max_scroll_per_post=args.max_scroll_post,
            target_rows=args.target,
            resume=not args.no_resume,
        )

    print(f"Output akhir: {scraper.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())