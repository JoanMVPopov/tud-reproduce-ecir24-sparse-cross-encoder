"""
Reproduce TREC DL Passage nDCG@10 for sparse-cross-encoder-4-512 (or manually fine-tuned version of it)
Generates BM25 baselines, re-ranks with a given model, and computes nDCG@10 micro and macro.

Usage:
    python evaluate_passage.py --model webis/sparse-cross-encoder-4-512
    python evaluate_passage.py --model your-hf-username/your-model --skip_bm25 --skip_rerank
"""

import argparse
import subprocess
import sys
from pathlib import Path

import ir_datasets
import pandas as pd
import trectools

DATASETS = [
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "msmarco-passage-v2/trec-dl-2021/judged",
    "msmarco-passage-v2/trec-dl-2022/judged",
    # "beir/hotpotqa/test", # if you decide to use these, see "Note" below
    # "beir/nq",
    # "beir/dbpedia-entity/test"
]

BASELINE_DIR = Path("data/baseline-runs/bm25")
RERANKED_DIR = Path("data/reranked-runs")
INDEX_DIR = Path("data/index")


def dataset_to_filename(dataset):
    return dataset.replace("/", "-") + ".run"


def generate_bm25_baselines():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    missing = [d for d in DATASETS if not (BASELINE_DIR / dataset_to_filename(d)).exists()]
    if not missing:
        print("All BM25 baselines already exist.")
        return
    print(f"Generating BM25 baselines for {len(missing)} datasets...")
    subprocess.run([
        sys.executable, "sparse_cross_encoder/data/create_baseline_run.py",
        "--searcher", "bm25",
        "--ir_datasets", *missing,
        "--run_dir", "data/baseline-runs",
        "--index_dir", str(INDEX_DIR),
        "--k", "100",
    ], check=True)


def rerank(model, batch_size):
    model_name = model.replace("/", "-")
    model_dir = RERANKED_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        filename = dataset_to_filename(dataset)
        baseline = BASELINE_DIR / filename
        output = model_dir / filename
        if output.exists():
            print(f"Already re-ranked: {filename}")
            continue
        if not baseline.exists():
            print(f"ERROR: missing baseline {baseline}")
            continue
        print(f"\nRe-ranking {dataset}...")
        subprocess.run([
            sys.executable, "main.py", "predict",
            "--config", "sparse_cross_encoder/configs/cli/predict.yaml",
            "--model.model_name_or_path", model,
            "--data.ir_dataset_path", str(baseline),
            "--data.init_args.batch_size", str(batch_size),
            "--trainer.callbacks.output_path", str(output),
        ], check=True)


def evaluate(model):
    results = {}
    for dataset in DATASETS:
        model_name = model.replace("/", "-")
        model_dir = RERANKED_DIR / model_name
        run_path = model_dir / dataset_to_filename(dataset)
        if not run_path.exists():
            print(f"WARNING: missing {run_path}")
            continue

        run_df = pd.read_csv(
            run_path, sep=r"\s+", header=None,
            names=["query", "q0", "docid", "rank", "score", "system"],
            dtype={"query": str, "docid": str},
        )
        run = trectools.TrecRun()
        run.run_data = run_df

        qrels_df = pd.DataFrame(ir_datasets.load(dataset).qrels_iter())
        qrels_df = qrels_df.rename(
            {"query_id": "query", "doc_id": "docid", "relevance": "rel", "iteration": "q0"},
            axis=1,
        ).astype({"query": str, "docid": str})
        qrels = trectools.TrecQrel()
        qrels.qrels_data = qrels_df

        ev = trectools.TrecEval(run, qrels)
        # results[dataset] = ev.get_ndcg(depth=10)
        per_query = ev.get_ndcg(depth=10, per_query=True)
        # get last column (actual ndcg scores)
        # fillna because some queries might have no relevant documents
        per_query = per_query.iloc[:, -1].fillna(0)
        results[dataset] = per_query

    print("\n" + "=" * 50)
    print("nDCG@10 Results (Passage)")
    print("=" * 50)

    all_queries = []
    macro_scores = []

    # Note - make sure to change the print logic based on the datasets you're using
    # e.g. using year and v2 below will not work on BEIR

    for dataset, scores in results.items():
        year = dataset.split("trec-dl-")[1].split("/")[0]
        v2 = "-v2" if "v2" in dataset else ""
        # print(f"  TREC DL {year}{v2:4s}  {score:.3f}")
        mean = scores.mean()
        macro_scores.append(mean)
        print(f"  TREC DL {year}{v2:4s}  {mean:.3f}")
        # convert scores series to list and extend the main list
        all_queries.extend(scores.tolist())

    if results:
        # avg = sum(results.values()) / len(results)
        # print(f"  {'Average':16s}  {avg:.3f}")
        macro = sum(macro_scores) / len(macro_scores)
        micro = sum(all_queries) / len(all_queries)
        print(f"  {'Macro avg':16s}  {macro:.3f}")
        print(f"  {'Micro avg':16s}  {micro:.3f}")
    print("=" * 50)
    print("Paper target: 0.612 (Sparse CE w=4, passage avg (micro))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--skip_rerank", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    if not args.skip_bm25:
        generate_bm25_baselines()
    if not args.skip_rerank:
        rerank(args.model, args.batch_size)
    if not args.skip_eval:
        evaluate()
