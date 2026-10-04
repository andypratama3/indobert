"""
Central path configuration.

Every downstream analysis script read `data/results/indobert_absa_result.csv` as a
hardcoded literal. Changing the prediction source therefore meant editing seven
separate files, and nothing flagged that the file being read was the contaminated
one. All paths now resolve through this module, so the prediction source is
declared in exactly one place.

The contaminated original is preserved as
`data/results/indobert_absa_result_leaky.csv` for audit purposes and must not be
used for any reported result.
"""

from pathlib import Path

# --- directories ---------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotation"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = Path("models")

# --- stage 0: raw scrape --------------------------------------------------
RAW_COMMENTS = RAW_DIR / "tiktok_comments_mobil_dinas_kaltim.csv"

# --- stage 1: corpus ------------------------------------------------------
COMMENTS_CLEAN = PROCESSED_DIR / "comments_clean.csv"
COMMENTS_RELEVANT = PROCESSED_DIR / "comments_relevant.csv"
COMMENTS_IRRELEVANT = PROCESSED_DIR / "comments_irrelevant.csv"

# --- stage 2: annotated data ---------------------------------------------
ANNOTATION_DATASET = ANNOTATION_DIR / "dataset_train.csv"
TRAIN_DATASET_CLEAN = PROCESSED_DIR / "dataset_train_clean.csv"

# --- stage 3: models ------------------------------------------------------
CV_MODEL_DIR = MODELS_DIR / "indobert_aspect_sentiment_cv"
N_SPLITS = 10
RANDOM_STATE = 42

# --- stage 4: prediction --------------------------------------------------
# Single source of truth for the inference output. Everything downstream reads
# this file. It is produced by `python -m indobert.predict_absa_oof`, which uses
# out-of-fold prediction so no row is scored by a model that trained on it.
ABSA_RESULT = RESULTS_DIR / "indobert_absa_result_oof.csv"

# Retained only so the contamination can be demonstrated. Do not analyse this.
ABSA_RESULT_LEAKY = RESULTS_DIR / "indobert_absa_result_leaky.csv"

# --- stage 5: downstream analysis ----------------------------------------
THEMATIC_DETAIL = RESULTS_DIR / "thematic_coding_result.csv"
THEMATIC_SUMMARY = RESULTS_DIR / "thematic_summary.csv"
IDENTIFIKASI_DIR = RESULTS_DIR / "identifikasi_permasalahan"
CV_RESULTS_DIR = RESULTS_DIR / "cv"
VISUALIZATION_DIR = RESULTS_DIR / "visualization"

# Base directory of the repository, for scripts that need absolute paths.
BASE_DIR = Path(__file__).resolve().parent