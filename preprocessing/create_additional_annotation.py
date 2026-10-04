import pandas as pd
from pathlib import Path

RAW_PATH = Path(
    "data/raw/tiktok_comments_mobil_dinas_kaltim.csv"
)

ANNOTATION_PATH = Path(
    "data/annotation/aspect_sentiment_annotation_final.csv"
)

OUTPUT_PATH = Path(
    "data/annotation/aspect_sentiment_annotation_additional.csv"
)


def main():

    print(
        "Loading data..."
    )

    raw_df = pd.read_csv(
        RAW_PATH
    )

    annotated_df = pd.read_csv(
        ANNOTATION_PATH
    )

    raw_df["comment_text"] = (
        raw_df["comment_text"]
        .astype(str)
        .str.strip()
    )

    annotated_df["original_text"] = (
        annotated_df["original_text"]
        .astype(str)
        .str.strip()
    )

    annotated_texts = set(
        annotated_df["original_text"]
    )

    additional_df = raw_df[
        ~raw_df["comment_text"]
        .isin(annotated_texts)
    ].copy()

    additional_df = (
        additional_df
        .drop_duplicates(
            subset=["comment_text"]
        )
        .sample(
            n=min(
                250,
                len(additional_df)
            ),
            random_state=42,
        )
    )

    result = pd.DataFrame({

        "id": range(
            1,
            len(additional_df) + 1
        ),

        "username":
            additional_df[
                "comment_username"
            ].values,

        "original_text":
            additional_df[
                "comment_text"
            ].values,

        "reply_url":
            additional_df[
                "post_url"
            ].values,

        "source_tiktok":
            additional_df[
                "post_url"
            ].values,

        "annotator_aspect": "",

        "ai_aspect": "",

        "researcher_aspect": "",

        "final_aspect": "",

        "annotator_sentiment": "",

        "ai_sentiment": "",

        "researcher_sentiment": "",

        "final_sentiment": "",

        "final_label": "",

        "notes": "",
    })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nBerhasil membuat:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"Jumlah data tambahan: {len(result)}"
    )


if __name__ == "__main__":
    main()