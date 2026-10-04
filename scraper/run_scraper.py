from scraper.scraper import TwitterScraper

QUERIES = [
    "mobil dinas gubernur kaltim",
    "mobil dinas 8,5m",
    "gubernur kaltim mobil dinas",
    "jaga marwah kaltim",
    "mobil dinas kaltim",
    "mobil dinas kaltim 8,5 miliar",
    "kritik mobil dinas kaltim",
    "kontroversi mobil dinas kaltim",
    "viral mobil dinas kaltim",
]

scraper = TwitterScraper()
scraper.run_multi_query(
    queries=QUERIES,
    max_seed_tweets_per_query=420,
    max_scroll_search=80,
    max_scroll_per_tweet=300,
    max_relevant_seed_per_query=140,
    target_comment_rows=2000,
    save_all_comments=True,
    save_only_relevant_comments=False,
    min_reply_count_seed=1,
)