import os
import re
import time
import random
import pandas as pd
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class TwitterScraper:
    def __init__(self):
        self.seed_posts = []
        self.data_comments = []
        self.data_comments_relevant = []
        self.output_dir = "data/raw"
        self.driver = None

    def setup_driver(self):
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)

    def login_manual(self, timeout=180):
        print("Silakan login manual ke X...")
        self.driver.get("https://x.com/i/flow/login")
        print(f"Kamu punya waktu {timeout} detik untuk login.")

        start = time.time()
        while time.time() - start < timeout:
            current_url = self.driver.current_url.lower()

            if "/i/flow/login" not in current_url:
                print("Login terdeteksi selesai.")
                time.sleep(5)
                return True

            time.sleep(2)

        print("Login belum selesai sampai timeout.")
        return False

    def clean_text(self, text):
        return " ".join(str(text).lower().split())

    def parse_count(self, value):
        """
        Ubah teks count X seperti:
        12 -> 12
        1.2K -> 1200
        3K -> 3000
        1M -> 1000000
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

    def topic_score(self, text):
        text = self.clean_text(text)

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
        text = self.clean_text(text)

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

    def get_metric_count(self, article, metric_testid):
        try:
            metric_elem = article.find_element(By.CSS_SELECTOR, f'[data-testid="{metric_testid}"]')
            raw_text = metric_elem.text.strip()
            return self.parse_count(raw_text)
        except Exception:
            return 0

    def search_tweets(self, query, max_tweets=120, scroll_loops=30):
        # Pakai search default/popular, jangan f=live
        url = f"https://x.com/search?q={quote(query)}&src=typed_query"
        self.driver.get(url)
        time.sleep(8)

        tweets = []
        seen_urls = set()
        no_new_rounds = 0

        for _ in range(scroll_loops):
            before_count = len(seen_urls)
            elements = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

            for el in elements:
                try:
                    time_elem = el.find_element(By.CSS_SELECTOR, "time")
                    link = time_elem.find_element(By.XPATH, "./..").get_attribute("href")
                except Exception:
                    continue

                if not link or "/status/" not in link or link in seen_urls:
                    continue

                try:
                    text = el.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]').text.strip()
                except Exception:
                    text = ""

                reply_count = self.get_metric_count(el, "reply")
                retweet_count = self.get_metric_count(el, "retweet")
                like_count = self.get_metric_count(el, "like")

                item = {
                    "tweet_url": link,
                    "tweet_text": text,
                    "reply_count": reply_count,
                    "retweet_count": retweet_count,
                    "like_count": like_count,
                    "topic_score": self.topic_score(text),
                    "is_relevant_seed": self.is_relevant(text),
                    "query_used": query,
                }

                tweets.append(item)
                seen_urls.add(link)

                if len(tweets) >= max_tweets:
                    break

            if len(tweets) >= max_tweets:
                break

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(3, 6))

            if len(seen_urls) == before_count:
                no_new_rounds += 1
            else:
                no_new_rounds = 0

            if no_new_rounds >= 3:
                break

        tweets = sorted(
            tweets,
            key=lambda x: (
                x["is_relevant_seed"],
                x["reply_count"] >= 10,
                x["reply_count"],
                x["topic_score"],
                x["like_count"],
            ),
            reverse=True,
        )

        return tweets

    def extract_reply_url(self, article):
        try:
            time_elem = article.find_element(By.CSS_SELECTOR, "time")
            return time_elem.find_element(By.XPATH, "./..").get_attribute("href")
        except Exception:
            return ""

    def extract_username_and_display_name(self, article):
        username = ""
        display_name = ""

        try:
            user_elem = article.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
            lines = [x.strip() for x in user_elem.text.split("\n") if x.strip()]

            if len(lines) >= 1:
                display_name = lines[0]

            for line in lines:
                if line.startswith("@"):
                    username = line
                    break
        except Exception:
            pass

        return username, display_name

    def scrape_replies(self, tweet_url, seed_text="", query_used="", max_scroll=100):
        self.driver.get(tweet_url)
        time.sleep(6)

        replies = []
        seen_items = set()
        no_new_rounds = 0
        last_total = 0

        for _ in range(max_scroll):
            articles = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

            for a in articles:
                try:
                    text = a.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]').text.strip()
                    if not text:
                        continue
                except Exception:
                    continue

                reply_url = self.extract_reply_url(a)

                # Skip parent tweet utama
                if reply_url and reply_url == tweet_url:
                    continue

                username, display_name = self.extract_username_and_display_name(a)

                # Skip kalau bukan reply valid
                if not reply_url:
                    continue

                item_key = f"{username}|{text}|{reply_url}"
                if item_key in seen_items:
                    continue
                seen_items.add(item_key)

                like_count = self.get_metric_count(a, "like")
                reply_count = self.get_metric_count(a, "reply")
                retweet_count = self.get_metric_count(a, "retweet")

                row = {
                    "username": username,
                    "display_name": display_name,
                    "text": text,
                    "reply_url": reply_url,
                    "source_tweet": tweet_url,
                    "seed_text": seed_text,
                    "query_used": query_used,
                    "like_count": like_count,
                    "reply_count": reply_count,
                    "retweet_count": retweet_count,
                    "topic_score": self.topic_score(text),
                    "is_relevant": self.is_relevant(text),
                }

                replies.append(row)

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(3, 6))

            current_total = len(replies)
            if current_total == last_total:
                no_new_rounds += 1
            else:
                no_new_rounds = 0

            last_total = current_total

            if no_new_rounds >= 4:
                print("  Scroll mentok, tidak ada komentar baru.")
                break

        return replies

    def save_csv(self, df, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"File disimpan ke: {filepath}")

    def run_multi_query(
        self,
        queries,
        max_seed_tweets_per_query=120,
        max_scroll_search=30,
        max_scroll_per_tweet=100,
        max_relevant_seed_per_query=40,
        target_comment_rows=500,
        save_all_comments=True,
        save_only_relevant_comments=False,
        min_reply_count_seed=5,
    ):
        self.setup_driver()

        login_ok = self.login_manual(timeout=180)
        if not login_ok:
            print("Login gagal. Program dihentikan.")
            self.driver.quit()
            return

        used_seed_urls = set()

        for q_idx, query in enumerate(queries, start=1):
            print("\n" + "=" * 80)
            print(f"QUERY [{q_idx}/{len(queries)}]")
            print(query)
            print("=" * 80)

            seed_tweets = self.search_tweets(
                query=query,
                max_tweets=max_seed_tweets_per_query,
                scroll_loops=max_scroll_search,
            )

            print(f"Total seed tweet ditemukan: {len(seed_tweets)}")

            relevant_seed_tweets = [
                t for t in seed_tweets
                if t["is_relevant_seed"]
                and t["tweet_url"] not in used_seed_urls
                and t["reply_count"] >= min_reply_count_seed
            ]

            # fallback kalau terlalu sedikit
            if len(relevant_seed_tweets) < 5:
                relevant_seed_tweets = [
                    t for t in seed_tweets
                    if t["is_relevant_seed"] and t["tweet_url"] not in used_seed_urls
                ]

            relevant_seed_tweets = relevant_seed_tweets[:max_relevant_seed_per_query]

            print(f"Seed tweet relevan yang dipakai: {len(relevant_seed_tweets)}")

            self.seed_posts.extend(relevant_seed_tweets)

            for i, item in enumerate(relevant_seed_tweets, start=1):
                tweet_url = item["tweet_url"]
                seed_text = item["tweet_text"]
                used_seed_urls.add(tweet_url)

                print(f"[Q{q_idx} - {i}/{len(relevant_seed_tweets)}] Ambil komentar dari:")
                print(f"  {tweet_url}")
                print(f"  Seed score: {item['topic_score']}")
                print(f"  Reply count seed: {item['reply_count']}")

                replies = self.scrape_replies(
                    tweet_url=tweet_url,
                    seed_text=seed_text,
                    query_used=query,
                    max_scroll=max_scroll_per_tweet,
                )

                if save_all_comments:
                    self.data_comments.extend(replies)

                if save_only_relevant_comments:
                    self.data_comments_relevant.extend([r for r in replies if r["is_relevant"]])

                df_comments_tmp = pd.DataFrame(self.data_comments)
                if not df_comments_tmp.empty:
                    df_comments_tmp = df_comments_tmp.drop_duplicates(
                        subset=["username", "text", "reply_url"]
                    )
                    self.data_comments = df_comments_tmp.to_dict("records")

                df_rel_tmp = pd.DataFrame(self.data_comments_relevant)
                if not df_rel_tmp.empty:
                    df_rel_tmp = df_rel_tmp.drop_duplicates(
                        subset=["username", "text", "reply_url"]
                    )
                    self.data_comments_relevant = df_rel_tmp.to_dict("records")

                print(f"  Total komentar unik: {len(self.data_comments)}")
                print(f"  Total komentar relevan unik: {len(self.data_comments_relevant)}")

                if len(self.data_comments) >= target_comment_rows:
                    print(f"Target {target_comment_rows} komentar tercapai.")
                    break

            if len(self.data_comments) >= target_comment_rows:
                break

        if not self.data_comments and not self.data_comments_relevant:
            print("Tidak ada komentar berhasil diambil.")
            self.driver.quit()
            return

        if self.seed_posts:
            df_seed = pd.DataFrame(self.seed_posts).drop_duplicates(subset=["tweet_url"])
            self.save_csv(df_seed, "seed_posts_by_keyword.csv")

        if self.data_comments:
            df_comments = pd.DataFrame(self.data_comments).drop_duplicates(
                subset=["username", "text", "reply_url"]
            )
            self.save_csv(df_comments, "comments_all_by_keyword.csv")
            print(f"Jumlah akhir semua komentar: {len(df_comments)}")

        if self.data_comments_relevant:
            df_relevant = pd.DataFrame(self.data_comments_relevant).drop_duplicates(
                subset=["username", "text", "reply_url"]
            )
            self.save_csv(df_relevant, "comments_relevant_by_keyword.csv")
            print(f"Jumlah akhir komentar relevan: {len(df_relevant)}")

        self.driver.quit()
        print("Scraping selesai.")