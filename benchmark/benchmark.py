#!/usr/bin/env python3
"""
Language Classification Benchmark using Hyperdimensional Computing
==================================================================
Replicates and extends the methodology from:

  Joshi A, Halseth JT, Kanerva P. "Language geometry using random indexing."
  Conceptual Structures for STEM Research and Education. ICCS 2016.

Each method builds character n-gram hypervectors, bundles them into per-language
prototype vectors, and classifies new text by cosine similarity.

Dataset
-------
  papluca/language-identification  (HuggingFace Datasets)
  Converted into a Polars DataFrame where each row represents an individual n-gram
  with its components spread across columns (e.g., pos_0, pos_1, pos_2).

Installation
------------
  pip install numpy scikit-learn psutil datasets polars pyarrow
  pip install torch-hd  
  pip install hdlib
"""

import gc
import os
import time
import tracemalloc
import argparse
from collections import defaultdict
from typing import Protocol

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────
DIMENSIONS = 10_000
NGRAM_SIZE = 3
RANDOM_SEED = 42
MAX_TRAIN_SAMPLES_PER_LANG = 500
MAX_TEST_SAMPLES_PER_LANG = 100
BATCH_SIZE = 50_000  # Cap chunk sizes to protect RAM from OOM exceptions
DATASET_FILE = "hdc_dataset.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────
class HDCClassifier(Protocol):
    def train(self, df_train: pl.DataFrame) -> None: ...
    def predict(self, df_test: pl.DataFrame) -> dict[int, str]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Polars DataFrame Generation
# ─────────────────────────────────────────────────────────────────────────────


def _build_polars_rows(
    texts: list[str], labels: list[str], split: str, start_id: int, n: int
):
    """Expands texts into a tabular list of dictionaries representing n-grams."""
    rows = []
    sample_id = start_id
    for text, lang in zip(texts, labels):
        text = text.lower()
        for i in range(len(text) - n + 1):
            row = {
                "Sample ID": sample_id,
                "Language": lang,
                "Split": split,
            }
            # Add columns for individual parts of the n-gram
            for j in range(n):
                row[f"pos_{j}"] = text[i + j]
            rows.append(row)
        sample_id += 1
    return rows, sample_id


def generate_and_save_dataset(max_train: int, max_test: int, n: int) -> pl.DataFrame:
    from datasets import load_dataset as hf_load

    print("  Fetching papluca/language-identification …")
    ds_train = hf_load("papluca/language-identification", split="train")
    ds_test = hf_load("papluca/language-identification", split="validation")

    def _collect(ds, max_per_lang):
        by_lang: dict[str, list[str]] = defaultdict(list)
        for item in ds:
            by_lang[item["labels"]].append(item["text"])
        texts, labels = [], []
        for lang, txts in sorted(by_lang.items()):
            chosen = txts[:max_per_lang]
            texts.extend(chosen)
            labels.extend([lang] * len(chosen))
        return texts, labels

    print("  Extracting raw samples …")
    train_t, train_l = _collect(ds_train, max_train)
    test_t, test_l = _collect(ds_test, max_test)

    print(f"  Expanding into n-grams (n={n}) and building Polars DataFrame …")
    train_rows, next_id = _build_polars_rows(train_t, train_l, "train", 0, n)
    test_rows, _ = _build_polars_rows(test_t, test_l, "test", next_id, n)

    df = pl.DataFrame(train_rows + test_rows)

    # Use a shared Polars Enum type for the n-gram columns to save memory.
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    unique_chars = (
        pl.concat([df[c] for c in pos_cols]).unique().drop_nulls().sort().to_list()
    )
    shared_enum = pl.Enum(unique_chars)
    df = df.with_columns(pl.col(pos_cols).cast(shared_enum))

    print(f"  Saving to {DATASET_FILE} for reproducibility …")
    df.write_parquet(DATASET_FILE)
    return df


def get_dataset(
    max_train: int = MAX_TRAIN_SAMPLES_PER_LANG,
    max_test: int = MAX_TEST_SAMPLES_PER_LANG,
    n: int = NGRAM_SIZE,
) -> pl.DataFrame:
    """Loads existing dataset if available, otherwise generates it."""
    if os.path.exists(DATASET_FILE):
        print(f"  Loading existing dataset from {DATASET_FILE} …")
        return pl.read_parquet(DATASET_FILE)
    return generate_and_save_dataset(max_train, max_test, n)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NumPy HDC  (Batched Vectorization)
# ─────────────────────────────────────────────────────────────────────────────


class NumPyHDC:
    def __init__(
        self,
        dimensions: int = DIMENSIONS,
        ngram_size: int = NGRAM_SIZE,
        seed: int = RANDOM_SEED,
    ):
        self.dim = dimensions
        self.n = ngram_size
        self.rng = np.random.default_rng(seed)
        self._codebook_matrix: np.ndarray | None = None
        self.profiles: dict[str, np.ndarray] = {}

    def _init_codebook(self, df: pl.DataFrame) -> None:
        if self._codebook_matrix is not None:
            return
        pos_cols = [c for c in df.columns if c.startswith("pos_")]
        if not pos_cols:
            return
        categories = df[pos_cols[0]].dtype.categories
        self._codebook_matrix = self.rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=(len(categories), self.dim)
        )

    def train(self, df_train: pl.DataFrame) -> None:
        self._init_codebook(df_train)

        langs = df_train["Language"].to_numpy()
        unique_langs = np.unique(langs)
        lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
        lang_ids = np.array([lang_to_idx[l] for l in langs], dtype=np.int32)

        lang_sums = np.zeros((len(unique_langs), self.dim), dtype=np.float32)
        lang_counts = np.zeros(len(unique_langs), dtype=np.int32)

        ngram_cols = [c for c in df_train.columns if c.startswith("pos_")]
        idx_cols = [df_train[c].to_physical().to_numpy() for c in ngram_cols]
        n_rows = len(df_train)

        # Vectorized processing in blocks to protect RAM
        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)

            b_idx0 = idx_cols[0][start:end]
            v_matrix = self._codebook_matrix[b_idx0].copy()
            for k in range(1, len(ngram_cols)):
                b_idx_k = idx_cols[k][start:end]
                v_matrix *= np.roll(self._codebook_matrix[b_idx_k], k, axis=1)

            b_lang_ids = lang_ids[start:end]
            np.add.at(lang_sums, b_lang_ids, v_matrix)
            np.add.at(lang_counts, b_lang_ids, 1)

        for idx, lang in enumerate(unique_langs):
            self.profiles[lang] = lang_sums[idx] / max(lang_counts[idx], 1)

    def predict(self, df_test: pl.DataFrame) -> dict[int, str]:
        self._init_codebook(df_test)

        sample_ids = df_test["Sample ID"].to_numpy()
        unique_sids = np.unique(sample_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(unique_sids)}
        sid_ids = np.array([sid_to_idx[sid] for sid in sample_ids], dtype=np.int32)

        sample_vectors = np.zeros((len(unique_sids), self.dim), dtype=np.float32)

        ngram_cols = [c for c in df_test.columns if c.startswith("pos_")]
        idx_cols = [df_test[c].to_physical().to_numpy() for c in ngram_cols]
        n_rows = len(df_test)

        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)

            b_idx0 = idx_cols[0][start:end]
            v_matrix = self._codebook_matrix[b_idx0].copy()
            for k in range(1, len(ngram_cols)):
                b_idx_k = idx_cols[k][start:end]
                v_matrix *= np.roll(self._codebook_matrix[b_idx_k], k, axis=1)

            b_sid_ids = sid_ids[start:end]
            np.add.at(sample_vectors, b_sid_ids, v_matrix)

        norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True) + 1e-10
        sample_vectors_n = sample_vectors / norms

        lang_order = list(self.profiles.keys())
        mat = np.stack([self.profiles[l] for l in lang_order])
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)

        sims = sample_vectors_n @ mat_n.T
        best_lang_indices = np.argmax(sims, axis=1)

        return {
            int(sid): lang_order[idx]
            for sid, idx in zip(unique_sids, best_lang_indices)
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Symbolar HDC
# ─────────────────────────────────────────────────────────────────────────────


class SymbolarHDC:
    def __init__(
        self,
        dimensions: int = DIMENSIONS,
        ngram_size: int = NGRAM_SIZE,
        seed: int = RANDOM_SEED,
    ):
        from symbolar import MultiplyAddPermute, MapUnnormalizedVector

        self._arch = MultiplyAddPermute(seed)
        self.dim = dimensions
        self.n = ngram_size
        self.profiles: dict[str, MapUnnormalizedVector] = {}
        self._storage = None

    def train(self, df_train: pl.DataFrame) -> None:
        from symbolar import MapStorage

        ngram_cols = [c for c in df_train.columns if c.startswith("pos_")]

        self._storage = MapStorage.from_dataframe(
            self._arch, self.dim, df_train.select(ngram_cols)
        )

        langs = df_train["Language"].unique().sort().to_list()
        lang_codes = df_train["Language"].cast(pl.Enum(langs)).to_physical().to_list()

        subset = self._storage.subset(df_train.select(ngram_cols))
        grouped = subset.bundle_dataset_temporal_grouped(lang_codes)
        self.profiles = {langs[code]: vector for code, vector in grouped.items()}

    def predict(self, df_test: pl.DataFrame) -> dict[int, str]:
        if self._storage is None:
            raise RuntimeError("Model must be trained before prediction.")

        ngram_cols = [c for c in df_test.columns if c.startswith("pos_")]
        sample_ids = df_test["Sample ID"].to_list()

        subset = self._storage.subset(df_test.select(ngram_cols))
        grouped = subset.bundle_dataset_temporal_grouped(sample_ids)
        return {sid: query.best_match(self.profiles) for sid, query in grouped.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. torchhd HDC (Batched Vectorization)
# ─────────────────────────────────────────────────────────────────────────────


class TorchhHDC:
    def __init__(
        self,
        dimensions: int = DIMENSIONS,
        ngram_size: int = NGRAM_SIZE,
        seed: int = RANDOM_SEED,
    ):
        import torchhd
        import torch

        self._hd = torchhd
        self._torch = torch
        self._torch.manual_seed(seed)
        self.dim = dimensions
        self.n = ngram_size
        self._codebook_matrix = None
        self.profiles: dict[str, object] = {}

    def _init_codebook(self, df: pl.DataFrame) -> None:
        if self._codebook_matrix is not None:
            return
        pos_cols = [c for c in df.columns if c.startswith("pos_")]
        if not pos_cols:
            return
        categories = df[pos_cols[0]].dtype.categories
        self._codebook_matrix = self._hd.random(
            len(categories), self.dim, "MAP", dtype=self._torch.float32
        )

    def train(self, df_train: pl.DataFrame) -> None:
        self._init_codebook(df_train)

        langs = df_train["Language"].to_numpy()
        unique_langs = np.unique(langs)
        lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
        lang_ids = self._torch.tensor(
            [lang_to_idx[l] for l in langs], dtype=self._torch.long
        )

        lang_sums = self._torch.zeros(
            (len(unique_langs), self.dim), dtype=self._torch.float32
        )

        ngram_cols = [c for c in df_train.columns if c.startswith("pos_")]
        idx_cols = [
            self._torch.from_numpy(df_train[c].to_physical().to_numpy()).long()
            for c in ngram_cols
        ]
        n_rows = len(df_train)

        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)

            b_idx0 = idx_cols[0][start:end]
            v_matrix = self._codebook_matrix[b_idx0].clone()
            for k in range(1, len(ngram_cols)):
                b_idx_k = idx_cols[k][start:end]
                v_matrix = self._hd.bind(
                    v_matrix,
                    self._torch.roll(self._codebook_matrix[b_idx_k], shifts=k, dims=1),
                )

            lang_sums.index_add_(0, lang_ids[start:end], v_matrix)

        for idx, lang in enumerate(unique_langs):
            self.profiles[lang] = lang_sums[idx]

    def predict(self, df_test: pl.DataFrame) -> dict[int, str]:
        self._init_codebook(df_test)

        sample_ids = df_test["Sample ID"].to_numpy()
        unique_sids = np.unique(sample_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(unique_sids)}
        sid_ids = self._torch.tensor(
            [sid_to_idx[sid] for sid in sample_ids], dtype=self._torch.long
        )

        sample_vectors = self._torch.zeros(
            (len(unique_sids), self.dim), dtype=self._torch.float32
        )

        ngram_cols = [c for c in df_test.columns if c.startswith("pos_")]
        idx_cols = [
            self._torch.from_numpy(df_test[c].to_physical().to_numpy()).long()
            for c in ngram_cols
        ]
        n_rows = len(df_test)

        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)

            b_idx0 = idx_cols[0][start:end]
            v_matrix = self._codebook_matrix[b_idx0].clone()
            for k in range(1, len(ngram_cols)):
                b_idx_k = idx_cols[k][start:end]
                v_matrix = self._hd.bind(
                    v_matrix,
                    self._torch.roll(self._codebook_matrix[b_idx_k], shifts=k, dims=1),
                )

            sample_vectors.index_add_(0, sid_ids[start:end], v_matrix)

        lang_order = list(self.profiles.keys())
        profile_mat = self._torch.stack([self.profiles[l] for l in lang_order])

        norm_samples = sample_vectors / (
            self._torch.linalg.norm(sample_vectors, dim=1, keepdim=True) + 1e-10
        )
        norm_profiles = profile_mat / (
            self._torch.linalg.norm(profile_mat, dim=1, keepdim=True) + 1e-10
        )
        sims = norm_samples @ norm_profiles.t()

        best_lang_indices = self._torch.argmax(sims, dim=1).cpu().numpy()
        return {
            int(sid): lang_order[idx]
            for sid, idx in zip(unique_sids, best_lang_indices)
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. hdlib HDC  (Batched Vectorization)
# ─────────────────────────────────────────────────────────────────────────────


class HdlibHDC:
    def __init__(
        self,
        dimensions: int = DIMENSIONS,
        ngram_size: int = NGRAM_SIZE,
        seed: int = RANDOM_SEED,
    ):
        from hdlib.space import Space

        np.random.seed(seed)
        self.dim = dimensions
        self.n = ngram_size
        # Fixed: Removed the unsupported 'seed' parameter from Space initialization
        self._space = Space(size=dimensions, vtype="bipolar")
        self._codebook_matrix = None
        self.profiles: dict[str, np.ndarray] = {}

    def _init_codebook(self, df: pl.DataFrame) -> None:
        if self._codebook_matrix is not None:
            return
        from hdlib.space import Vector

        pos_cols = [c for c in df.columns if c.startswith("pos_")]
        if not pos_cols:
            return
        categories = df[pos_cols[0]].dtype.categories

        for ch in categories:
            key = f"c{ord(ch):05d}"
            # Space.insert() takes a Vector object, not a bare name.
            self._space.insert(Vector(name=key, size=self.dim, vtype="bipolar"))

        self._codebook_matrix = np.zeros((len(categories), self.dim), dtype=np.float32)
        for idx, ch in enumerate(categories):
            key = f"c{ord(ch):05d}"
            # Space has no __getitem__; look vectors up by name via get().
            self._codebook_matrix[idx] = self._space.get(names=[key])[0].vector.astype(
                np.float32
            )

    def train(self, df_train: pl.DataFrame) -> None:
        self._init_codebook(df_train)

        langs = df_train["Language"].to_numpy()
        unique_langs = np.unique(langs)
        lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
        lang_ids = np.array([lang_to_idx[l] for l in langs], dtype=np.int32)

        lang_sums = np.zeros((len(unique_langs), self.dim), dtype=np.float32)
        lang_counts = np.zeros(len(unique_langs), dtype=np.int32)

        ngram_cols = [c for c in df_train.columns if c.startswith("pos_")]
        idx_cols = [df_train[c].to_physical().to_numpy() for c in ngram_cols]
        n_rows = len(df_train)

        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)

            b_idx0 = idx_cols[0][start:end]
            v_matrix = self._codebook_matrix[b_idx0].copy()
            for k in range(1, len(ngram_cols)):
                b_idx_k = idx_cols[k][start:end]
                v_matrix *= np.roll(self._codebook_matrix[b_idx_k], k, axis=1)

            b_lang_ids = lang_ids[start:end]
            np.add.at(lang_sums, b_lang_ids, v_matrix)
            np.add.at(lang_counts, b_lang_ids, 1)

        for idx, lang in enumerate(unique_langs):
            self.profiles[lang] = lang_sums[idx] / max(lang_counts[idx], 1)

    def predict(self, df_test: pl.DataFrame) -> dict[int, str]:
        self._init_codebook(df_test)

        sample_ids = df_test["Sample ID"].to_numpy()
        unique_sids = np.unique(sample_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(unique_sids)}
        sid_ids = np.array([sid_to_idx[sid] for sid in sample_ids], dtype=np.int32)

        sample_vectors = np.zeros((len(unique_sids), self.dim), dtype=np.float32)

        ngram_cols = [c for c in df_test.columns if c.startswith("pos_")]
        idx_cols = [df_test[c].to_physical().to_numpy() for c in ngram_cols]
        n_rows = len(df_test)

        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)

            b_idx0 = idx_cols[0][start:end]
            v_matrix = self._codebook_matrix[b_idx0].copy()
            for k in range(1, len(ngram_cols)):
                b_idx_k = idx_cols[k][start:end]
                v_matrix *= np.roll(self._codebook_matrix[b_idx_k], k, axis=1)

            b_sid_ids = sid_ids[start:end]
            np.add.at(sample_vectors, b_sid_ids, v_matrix)

        norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True) + 1e-10
        sample_vectors_n = sample_vectors / norms

        lang_order = list(self.profiles.keys())
        mat = np.stack([self.profiles[l] for l in lang_order])
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)

        sims = sample_vectors_n @ mat_n.T
        best_lang_indices = np.argmax(sims, axis=1)

        return {
            int(sid): lang_order[idx]
            for sid, idx in zip(unique_sids, best_lang_indices)
        }

def _peak_mb_during(fn):
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return result, peak / 1_048_576


def benchmark(
    name: str, model: HDCClassifier, df_train: pl.DataFrame, df_test: pl.DataFrame
) -> dict:
    print(f"\n  [{name}]")

    # Training
    _, train_mb = _peak_mb_during(lambda: model.train(df_train))
    t0 = time.perf_counter()
    model.train(df_train)
    train_s = time.perf_counter() - t0
    print(f"    train  : {train_s:.2f} s  |  peak Δmem: {train_mb:.1f} MB")

    # Inference
    t0 = time.perf_counter()
    preds_dict = model.predict(df_test)
    infer_s = time.perf_counter() - t0

    # Calculate Accuracy
    test_samples = df_test.select(["Sample ID", "Language"]).unique()
    y_true, y_pred = [], []
    for row in test_samples.iter_rows(named=True):
        sid = row["Sample ID"]
        if sid in preds_dict:
            y_true.append(row["Language"])
            y_pred.append(preds_dict[sid])

    ms_per = infer_s / max(len(y_pred), 1) * 1000
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"    infer  : {infer_s:.2f} s  ({ms_per:.2f} ms/sample)")
    print(f"    acc    : {acc:.2f}%")

    return dict(
        model=name,
        accuracy=round(acc, 2),
        train_time_s=round(train_s, 2),
        infer_ms=round(ms_per, 2),
        peak_mem_mb=round(train_mb, 1),
    )


def print_table(rows: list[dict]) -> None:
    cols = ["model", "accuracy", "train_time_s", "infer_ms", "peak_mem_mb"]
    headers = ["Model", "Accuracy (%)", "Train (s)", "Infer (ms/samp)", "Peak Mem (MB)"]
    widths = [
        max(len(h), max(len(str(r[c])) for r in rows)) for h, c in zip(headers, cols)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:{w}}} " for w in widths) + "|"

    print("\n" + sep)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*[str(r[c]) for c in cols]))
    print(sep + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="HDC Language Classification Benchmark")
    p.add_argument("--dim", type=int, default=DIMENSIONS, help="Hypervector dimension")
    p.add_argument("--ngram", type=int, default=NGRAM_SIZE, help="N-gram size")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    p.add_argument(
        "--train",
        type=int,
        default=MAX_TRAIN_SAMPLES_PER_LANG,
        help="Max train samples per language",
    )
    p.add_argument(
        "--test",
        type=int,
        default=MAX_TEST_SAMPLES_PER_LANG,
        help="Max test samples per language",
    )
    return p.parse_args()


def main():
    args = parse_args()
    dim, ngram, seed = args.dim, args.ngram, args.seed

    print("=" * 64)
    print("  HDC Language Classification Benchmark")
    print("  Methodology: Language Geometry using Random Indexing")
    print("=" * 64)
    print(f"\n  dim={dim:,}  |  n-gram={ngram}  |  seed={seed}")

    print("\nLoading and formatting dataset …")
    df = get_dataset(args.train, args.test, ngram)
    df_train = df.filter(pl.col("Split") == "train")
    df_test = df.filter(pl.col("Split") == "test")
    langs = df["Language"].unique().to_list()

    unique_train_samples = df_train["Sample ID"].n_unique()
    unique_test_samples = df_test["Sample ID"].n_unique()
    print(f"  Languages ({len(langs)}): {', '.join(langs)}")
    print(
        f"  Train: {unique_train_samples} samples  |  Test: {unique_test_samples} samples"
    )

    candidates = [
        ("NumPy HDC", NumPyHDC, {}),
        ("Symbolar", SymbolarHDC, {}),
        ("torchhd", TorchhHDC, {}),
        ("hdlib", HdlibHDC, {}),
    ]

    print("\nRunning benchmarks …")
    results = []
    for name, cls, kwargs in candidates:
        try:
            model = cls(dim, ngram, seed, **kwargs)
            r = benchmark(name, model, df_train, df_test)
            results.append(r)
        except ImportError as e:
            print(f"\n  [{name}]  ✗ skipped – {e}")
        except Exception as e:
            print(f"\n  [{name}]  ✗ error   – {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print_table(results)


if __name__ == "__main__":
    main()
