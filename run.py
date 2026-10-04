from scraper.scraper import TwitterScraper
import pandas as pd
import time
import os

QUERIES = [
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

    # Opsi 2 — query baru variasi
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

SAVE_PATH_COMMENTS = "data/raw/comments_all_by_keyword.csv"
SAVE_PATH_SEED = "data/raw/seed_posts_by_keyword.csv"

def load_existing_urls(path, col):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if col in df.columns:
            return set(df[col].tolist())
    return set()

def save_progress(scraper):
    if scraper.data_comments:
        df = pd.DataFrame(scraper.data_comments).drop_duplicates(
            subset=["username", "text", "reply_url"]
        )
        if os.path.exists(SAVE_PATH_COMMENTS):
            df_old = pd.read_csv(SAVE_PATH_COMMENTS)
            df = pd.concat([df_old, df]).drop_duplicates(
                subset=["username", "text", "reply_url"]
            )
        df.to_csv(SAVE_PATH_COMMENTS, index=False)
        print(f"  [AUTO-SAVE] {len(df)} komentar tersimpan.")

    if scraper.seed_posts:
        df_seed = pd.DataFrame(scraper.seed_posts).drop_duplicates(subset=["tweet_url"])
        if os.path.exists(SAVE_PATH_SEED):
            df_old = pd.read_csv(SAVE_PATH_SEED)
            df_seed = pd.concat([df_old, df_seed]).drop_duplicates(subset=["tweet_url"])
        df_seed.to_csv(SAVE_PATH_SEED, index=False)

MAX_RETRIES = 99

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"\n=== ATTEMPT {attempt} ===")

        already_scraped = load_existing_urls(SAVE_PATH_COMMENTS, "reply_url")
        already_seed = load_existing_urls(SAVE_PATH_SEED, "tweet_url")
        print(f"Komentar sudah ada: {len(already_scraped)}")
        print(f"Seed sudah ada: {len(already_seed)}")

        # Cek total sekarang
        if os.path.exists(SAVE_PATH_COMMENTS):
            total_now = len(pd.read_csv(SAVE_PATH_COMMENTS))
            print(f"Total komentar saat ini: {total_now}")
            if total_now >= 5000:
                print("TARGET 5000 SUDAH TERCAPAI!")
                exit()

        scraper = TwitterScraper()
        scraper.setup_driver()
        login_ok = scraper.login_manual(timeout=180)
        if not login_ok:
            print("Login gagal.")
            break

        scraper.used_seed_urls = set()  # fresh start biar seed lama bisa diproses ulang

        for q_idx, query in enumerate(QUERIES, start=1):
            print(f"\n{'='*60}")
            print(f"QUERY [{q_idx}/{len(QUERIES)}]: {query}")
            print('='*60)

            seed_tweets = scraper.search_tweets(
                query=query,
                max_tweets=600,
                scroll_loops=150,
            )

            # Opsi 1 — hapus filter is_relevant_seed, ambil semua tweet
            relevant_seeds = [
                t for t in seed_tweets
                if t["tweet_url"] not in scraper.used_seed_urls
                and t["tweet_url"] not in already_seed  # skip yang sudah pernah di-scrape
            ][:200]

            print(f"Seed baru ditemukan: {len(relevant_seeds)}")
            scraper.seed_posts.extend(relevant_seeds)

            for i, item in enumerate(relevant_seeds, start=1):
                tweet_url = item["tweet_url"]
                scraper.used_seed_urls.add(tweet_url)

                print(f"[Q{q_idx}-{i}/{len(relevant_seeds)}] {tweet_url}")
                print(f"  Reply count: {item['reply_count']}")

                replies = scraper.scrape_replies(
                    tweet_url=tweet_url,
                    seed_text=item["tweet_text"],
                    query_used=query,
                    max_scroll=500,
                )
                scraper.data_comments.extend(replies)

                # Auto-save setiap selesai 1 tweet
                save_progress(scraper)
                scraper.data_comments = []
                scraper.seed_posts = []

                # Cek total tersimpan
                if os.path.exists(SAVE_PATH_COMMENTS):
                    total = len(pd.read_csv(SAVE_PATH_COMMENTS))
                    print(f"  Total tersimpan: {total}")
                    if total >= 5000:
                        print("TARGET 5000 TERCAPAI!")
                        scraper.driver.quit()
                        exit()

        scraper.driver.quit()
        print("Semua query selesai!")
        break

    except Exception as e:
        print(f"\nCRASH: {e}")
        try:
            save_progress(scraper)
            scraper.driver.quit()
        except:
            pass
        print("Tunggu 30 detik lalu restart...")
        time.sleep(30)