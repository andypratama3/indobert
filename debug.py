import pandas as pd

df = pd.read_csv("data/raw/comments_all_by_keyword.csv")
df_seed = pd.read_csv("data/raw/seed_posts_by_keyword.csv")

# Tweet seed yang reply_count tinggi tapi komentar yang keambil sedikit
print("=== SEED DENGAN REPLY TERBANYAK ===")
top_seed = df_seed.sort_values("reply_count", ascending=False).head(10)
print(top_seed[["tweet_url", "reply_count", "query_used"]].to_string())

print("\n=== BERAPA KOMENTAR YANG KEAMBIL PER SEED TWEET ===")
comment_per_seed = df.groupby("source_tweet").size().reset_index(name="komen_diambil")
merged = top_seed.merge(comment_per_seed, left_on="tweet_url", right_on="source_tweet", how="left")
merged["komen_diambil"] = merged["komen_diambil"].fillna(0).astype(int)
print(merged[["tweet_url", "reply_count", "komen_diambil", "query_used"]].to_string())