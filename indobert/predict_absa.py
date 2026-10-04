"""
DEPRECATED -- DO NOT RUN. Retained only as an audit artefact.

This module reproduces BUG-01. It loaded a single fold model
(`models/indobert_aspect_sentiment_cv/fold_6`) and predicted every row of the
corpus. Because the corpus is a superset of the annotated training set, 673 of
the 998 rows it scored (67.4%) had been seen by that model during fine-tuning.

The output it produced, `data/results/indobert_absa_result.csv`, was the
contaminated input that every downstream analysis script originally consumed. It
reported 97.19% exact-label accuracy against a true figure near 78%, and that
overstatement propagated into every table and figure under `data/results/`.

The original implementation is preserved in git history at commit 876dc00 and its
predecessor 54c98db.

USE INSTEAD
-----------
    python -m indobert.predict_absa_oof

That performs out-of-fold prediction: each annotated row is scored by the one
fold model that held it out, and never-annotated rows by a ten-fold majority
vote. No row is scored by a model that trained on it.

Nothing in the pipeline imports this module. It is safe to delete once you no
longer need it as a reference.
"""

import sys

MESSAGE = """\
ABORTED: indobert/predict_absa.py is deprecated.

It leaks training data into inference (BUG-01): 673 of 998 predicted rows, 67.4%,
had been seen by the model during fine-tuning. Every number it produced is
unusable.

Run instead:
    python -m indobert.predict_absa_oof
"""


def main() -> int:
    print(MESSAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())