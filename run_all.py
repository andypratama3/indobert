"""
Single-command runner for the whole pipeline.

    python run_all.py                # everything except training
    python run_all.py --list         # show stages
    python run_all.py --include-train# also retrain (hours)
    python run_all.py --only 6 7     # run selected stages
    python run_all.py --from 6       # resume from a stage
    python run_all.py --cleanup      # delete stale checkpoints first

Or just double-click `run_all.bat`.

DESIGN NOTES
------------
Training is excluded by default. `indobert.train_cv` writes ~80 GB and takes
hours, and it is only necessary if the annotated dataset changed. Every other
stage is seconds to a couple of minutes, so a full verification pass over an
existing set of checkpoints costs about four minutes.

Each stage declares the files it must produce. After a stage runs, those files
are checked, so a stage that exits 0 but silently writes nothing is still
reported as a failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parent
VENV_PY = BASE / ".venv" / "Scripts" / "python.exe"

# Stages that run by default.
DEFAULT_ON = True
# Stage that is opt-in because it is slow and rarely needed.
TRAIN_OPTIONAL = False


@dataclass
class Stage:
    num: int
    key: str
    title: str
    module: str | None = None
    manual: bool = False
    default_on: bool = DEFAULT_ON
    optional: bool = False
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    note: str = ""


STAGES: list[Stage] = [
    Stage(
        0, "env", "Environment check", manual=True,
        inputs=["requirements.txt"],
        note="Confirms the virtualenv and required packages.",
    ),
    Stage(
        1, "corpus", "Build analysable corpus",
        module="preprocessing.preprocess",
        inputs=["data/raw/tiktok_comments_mobil_dinas_kaltim.csv"],
        outputs=["data/processed/comments_clean.csv"],
        note="Strips TikTok date stamps, user-aware dedup. 1081 raw -> 1031 usable.",
    ),
    Stage(
        2, "trainset", "Clean training data",
        module="preprocessing.preprocess_absa",
        inputs=["data/annotation/dataset_train.csv"],
        outputs=["data/processed/dataset_train_clean.csv"],
        note="Idempotent. Asserts the stored column reproduces 800/800.",
    ),
    Stage(
        3, "verify_train", "Verify training data integrity", manual=True,
        inputs=["data/processed/dataset_train_clean.csv"],
        note="Hard gate. Refuses to continue if the cleaner has drifted from the "
             "data the checkpoints were trained on (BUG-05 / BUG-12).",
    ),
    Stage(
        4, "annotate", "Manual annotation", manual=True,
        inputs=["data/annotation/README.md"],
        note="HUMAN STEP. Cannot be automated. The runner only checks that "
             "dataset_train.csv exists and has the required columns.",
    ),
    Stage(
        5, "train", "Fine-tune 10-fold (SLOW, hours, ~80GB)",
        module="indobert.train_cv",
        default_on=TRAIN_OPTIONAL, optional=True,
        inputs=["data/processed/dataset_train_clean.csv"],
        outputs=[
            "data/results/cv/cv_per_fold_metrics.csv",
            "data/results/cv/cv_summary_metrics.csv",
        ],
        note="Only needed if the annotated set changed. Excluded by default.",
    ),
    Stage(
        6, "predict", "Leakage-free out-of-fold prediction",
        module="indobert.predict_absa_oof",
        inputs=[
            "data/processed/comments_clean.csv",
            "data/processed/dataset_train_clean.csv",
            "models/indobert_aspect_sentiment_cv/fold_1/model.safetensors",
        ],
        outputs=["data/results/indobert_absa_result_oof.csv"],
        note="Every row scored by a model that never trained on it (BUG-01).",
    ),
    Stage(
        7, "thematic", "Thematic coding", module="thematic_coding",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/thematic_coding_result.csv",
            "data/results/thematic_summary.csv",
        ],
    ),
    Stage(
        8, "thematic_plot", "Thematic distribution figure",
        module="plot_thematic_distribution",
        inputs=["data/results/thematic_summary.csv"],
        outputs=[
            "data/results/visualization/gambar_5_10_distribusi_tema.png"
        ],
    ),
    Stage(
        9, "issue_akuntabilitas", "Issue identification: Akuntabilitas",
        module="identifikasi_masalah_akuntabilitas",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/bukti_komentar_akuntabilitas.csv",
            "data/results/rekap_masalah_akuntabilitas.csv",
        ],
    ),
    Stage(
        10, "issue_efektivitas", "Issue identification: Efektivitas & Efisiensi",
        module="identifikasi_masalah_efektivitas_efisiensi",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/bukti_komentar_efektivitas_efisiensi.csv",
            "data/results/rekap_masalah_efektivitas_efisiensi.csv",
        ],
    ),
    Stage(
        11, "issue_responsivitas", "Issue identification: Responsivitas",
        module="identifikasi_masalah_responsivitas",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/bukti_komentar_responsivitas.csv",
            "data/results/rekap_masalah_responsivitas.csv",
        ],
    ),
    Stage(
        12, "issue_transparansi", "Issue identification: Transparansi",
        module="identifikasi_masalah_transparansi",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/bukti_komentar_transparansi.csv",
            "data/results/rekap_masalah_transparansi.csv",
        ],
    ),
    Stage(
        13, "rca", "RCA and recommendations", module="analysis.analysis_pipeline",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/identifikasi_permasalahan/rca_rekomendasi.csv",
            "data/results/identifikasi_permasalahan/laporan_lengkap.csv",
            "data/results/identifikasi_permasalahan/sampel_komentar_negatif.csv",
            "data/results/identifikasi_permasalahan/distribusi_negatif_per_aspek.png",
        ],
    ),
    Stage(
        14, "provenance", "Provenance and sampling-bias audit",
        module="analysis.provenance_audit",
        inputs=[
            "data/annotation/dataset_train.csv",
            "data/raw/tiktok_comments_mobil_dinas_kaltim.csv",
        ],
        outputs=[
            "data/results/provenance_audit.csv",
            "data/results/provenance_missing_rows.csv",
            "data/annotation/annotated_not_in_corpus.csv",
        ],
        note="Quantifies the 116 annotated rows absent from the raw corpus (BUG-07).",
    ),
    Stage(
        15, "imbalance", "Class-imbalance and per-class metrics",
        module="analysis.class_imbalance_report",
        inputs=[
            "data/results/cv/cv_per_fold_metrics.csv",
            "data/results/cv/confusion_matrix_fold_1.csv",
        ],
        outputs=[
            "data/results/cv/per_class_metrics.csv",
            "data/results/cv/fold_class_coverage.csv",
        ],
        note="Reveals that three positive classes have F1 = 0.000 (BUG-08).",
    ),
    Stage(
        16, "viz", "Figures", module="visualization.visualize_absa",
        inputs=["data/results/indobert_absa_result_oof.csv"],
        outputs=[
            "data/results/visualization/aspect_sentiment_distribution.png",
            "data/results/visualization/distribusi_aspek.png",
            "data/results/visualization/distribusi_sentimen.png",
        ],
    ),
    Stage(
        17, "viz_cm", "Confusion matrix figures",
        module="visualization.plot_confusion_matrix",
        inputs=["data/results/cv"],
        outputs=[
            "data/results/visualization/confusion_matrix_fold_1.png",
            "data/results/visualization/confusion_matrix_fold_10.png",
        ],
    ),
]

REQUIRED_PKGS = [
    "torch", "transformers", "accelerate", "sklearn",
    "pandas", "numpy", "matplotlib", "selenium",
]

ANNOTATION_REQUIRED_COLS = [
    "original_text", "final_aspect", "final_sentiment", "final_label",
]


class Fail(Exception):
    pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def pick_python() -> str:
    """Prefer the project virtualenv, re-execing into it if necessary."""
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def hr(char="-", width=74):
    print(char * width)


def stage_enabled(s: Stage, args) -> tuple[bool, str]:
    if args.only:
        return (s.key in args.only or str(s.num) in args.only), "selected"
    if args.skip and (s.key in args.skip or str(s.num) in args.skip):
        return False, "skipped"
    if args.from_stage is not None and s.num < args.from_stage:
        return False, "before --from"
    if args.to_stage is not None and s.num > args.to_stage:
        return False, "after --to"
    if s.optional and not args.include_train:
        return False, "opt-in (--include-train)"
    if not s.default_on:
        return False, "off by default"
    return True, ""


# --------------------------------------------------------------------------- #
# built-in stages
# --------------------------------------------------------------------------- #

def run_env_check() -> None:
    import importlib

    print(f"Python      : {sys.version.split()[0]}")
    print(f"Interpreter : {sys.executable}")
    if not VENV_PY.exists():
        print(f"  WARNING   : {VENV_PY.name} not found, using the ambient interpreter.")
    print("Packages:")
    missing = []
    for p in REQUIRED_PKGS:
        try:
            mod = importlib.import_module(p)
            print(f"  {p:14} {getattr(mod, '__version__', 'n/a')}")
        except ImportError:
            missing.append(p)
    if missing:
        raise Fail(
            "missing packages: " + ", ".join(missing)
            + "\n           fix with: pip install -r requirements.txt"
        )
    try:
        import torch
        print(f"CUDA        : {torch.cuda.is_available()}")
    except Exception:
        pass


def run_verify_training_data() -> None:
    import pandas as pd

    from preprocessing.text_cleaning import normalize

    df = pd.read_csv("data/processed/dataset_train_clean.csv")
    n = len(df)
    ok = int((df["original_text"].astype(str).map(normalize) == df["clean_text"]).sum())
    print(f"rows                  : {n}")
    print(f"cleaner reproduces    : {ok}/{n}")
    if ok != n:
        raise Fail(
            f"Only {ok}/{n} rows reproduce. The cleaner has drifted from the data "
            "the checkpoints were trained on. Do NOT continue -- predictions "
            "would be made on unfamiliar text (BUG-05 / BUG-12)."
        )
    print("OK: training data is intact.")


def run_annotation_gate() -> None:
    import pandas as pd

    p = Path("data/annotation/dataset_train.csv")
    if not p.exists():
        raise Fail(
            "data/annotation/dataset_train.csv is missing. Annotation is a manual "
            "step -- see data/annotation/README.md and RUN_GUIDE.md step 2."
        )
    df = pd.read_csv(p)
    print(f"rows: {len(df)}")
    missing = [c for c in ANNOTATION_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise Fail("missing columns: " + ", ".join(missing))
    print("columns: OK")
    print("distribusi final_sentiment:")
    print(df["final_sentiment"].value_counts().to_string())


def run_cleanup_checkpoints() -> None:
    import shutil

    root = Path("models")
    ckpts = list(root.rglob("checkpoint-*"))
    if not ckpts:
        print("no checkpoint-* directories found.")
        return
    total = 0
    for c in ckpts:
        for f in c.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    print(f"found {len(ckpts)} checkpoint directories, {total / 1e9:.1f} GB")
    print("Each fold keeps its final model.safetensors, so these hold only "
          "intermediate weights and optimiser state.")
    if not shutil.disk_usage(BASE).free > total * 1.5:
        pass
    shutil.rmtree  # noqa: B018  (reference to make the intent explicit)
    for c in ckpts:
        shutil.rmtree(c, ignore_errors=True)
    print(f"deleted {len(ckpts)} directories, reclaimed {total / 1e9:.1f} GB")


MANUAL = {
    "env": run_env_check,
    "verify_train": run_verify_training_data,
    "annotate": run_annotation_gate,
}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def cmd_list() -> None:
    hr("=")
    print("PIPELINE STAGES")
    hr("=")
    for s in STAGES:
        flag = "opt-in" if s.optional else ("manual" if s.manual else "auto")
        print(f"  {s.num:2}. [{flag:6}] {s.title}")
        print(f"        module: {s.module or MANUAL.get(s.key, lambda: None).__name__}")
        if s.note:
            print(f"        note:   {s.note}")
    hr("=")
    print("default run  : python run_all.py")
    print("with training: python run_all.py --include-train")
    print("single stage : python run_all.py --only predict")
    hr("=")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the whole pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--include-train", action="store_true",
                    help="include the slow 10-fold retraining stage")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="run only these stages (key or number)")
    ap.add_argument("--skip", nargs="+", metavar="KEY", help="skip these stages")
    ap.add_argument("--from", dest="from_stage", type=int, help="start at this stage number")
    ap.add_argument("--to", dest="to_stage", type=int, help="stop at this stage number")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete stale checkpoint-* directories before running (destructive)")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing stage instead of stopping")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return 0

    py = pick_python()
    if Path(py).resolve() != Path(sys.executable).resolve():
        print(f"Re-executing with {py}\n")

    hr("=")
    print("INDOBERT ABSA PIPELINE")
    hr("=")
    print(f"root       : {BASE}")
    print(f"python     : {py}")
    print(f"started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hr("=")

    if args.cleanup:
        print("\n>>> PRE-FLIGHT: removing stale checkpoints\n")
        run_cleanup_checkpoints()

    results: list[tuple[Stage, str, float, str]] = []
    failures = 0

    for s in STAGES:
        enabled, why = stage_enabled(s, args)
        if not enabled:
            print(f"[{s.num:2}] SKIP  {s.title}  ({why})")
            results.append((s, "SKIP", 0.0, why))
            continue

        hr()
        print(f"[{s.num:2}] {s.title}")
        print(f"     module : {s.module or 'built-in check'}")
        hr()

        missing_in = [p for p in s.inputs if not (BASE / p).exists()]
        if missing_in:
            msg = "missing required input: " + ", ".join(missing_in)
            print(f"     FAIL  {msg}")
            failures += 1
            results.append((s, "FAIL", 0.0, msg))
            if not args.keep_going:
                print("\nAborting. Use --keep-going to continue past failures.")
                break
            continue

        t0 = time.time()
        try:
            if s.manual:
                MANUAL[s.key]()
            else:
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                proc = subprocess.run(
                    [py, "-m", s.module],
                    cwd=BASE, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                if proc.returncode != 0:
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                    for line in tail[-15:]:
                        print("     | " + line)
                    raise Fail(f"exit code {proc.returncode}")
            dt = time.time() - t0
        except Fail as e:
            dt = time.time() - t0
            print(f"     FAIL  {e}")
            failures += 1
            results.append((s, "FAIL", dt, str(e)))
            if not args.keep_going:
                print("\nAborting. Use --keep-going to continue past failures.")
                break
            continue
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            print(f"     ERROR {type(e).__name__}: {e}")
            failures += 1
            results.append((s, "FAIL", dt, f"{type(e).__name__}: {e}"))
            if not args.keep_going:
                break
            continue

        absent = [p for p in s.outputs if not (BASE / p).exists()]
        if absent:
            msg = "stage exited 0 but did not produce: " + ", ".join(absent)
            print(f"     FAIL  {msg}")
            failures += 1
            results.append((s, "FAIL", dt, msg))
            if not args.keep_going:
                break
            continue

        print(f"     OK    {dt:.1f}s")
        results.append((s, "OK", dt, ""))

    hr("=")
    print("SUMMARY")
    hr("=")
    for s, status, dt, why in results:
        colour = {"OK": "OK  ", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        line = f"  {colour}  [{s.num:2}] {s.title}"
        if status == "OK":
            line += f"  ({dt:.1f}s)"
        elif why:
            line += f"  ({why})"
        print(line)

    ran = [r for r in results if r[1] == "OK"]
    total = sum(r[2] for r in results)
    hr("=")
    print(f"  {len(ran)}/{len(STAGES)} stages succeeded in {total:.1f}s")

    if failures:
        print(f"  {failures} FAILED -- see docs/BUG_TRACKER.md")
        hr("=")
        return 1

    print("\n  Outputs are in data/results/. Read docs/RESULTS_FIXED.md before")
    print("  quoting any number, and docs/BUG_TRACKER.md for the caveats that")
    print("  must accompany them.")
    hr("=")
    return 0


if __name__ == "__main__":
    sys.exit(main())