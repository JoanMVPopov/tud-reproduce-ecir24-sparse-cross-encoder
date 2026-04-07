"""
Paired TOST (Two One-Sided Tests) for equivalence between cross-encoder models
on datasets. Tests whether nDCG@10 differences are within ±epsilon.

Compares each model against a baseline. Uses Bonferroni correction.


Usage:
    python tost_equivalence.py \
        --baseline "Full Attention" data/reranked-runs-full \
        --models \
            "Your Sparse" data/reranked-runs-your-sparse \
            "Their Sparse" data/reranked-runs-their-sparse \
        --epsilon 0.02
"""

import argparse
from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd
import trectools
from statsmodels.stats.weightstats import ttost_paired

DATASETS = [
    "beir/hotpotqa/test",
    "beir/nq",
    "beir/dbpedia-entity/test",
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "msmarco-passage-v2/trec-dl-2021/judged",
    "msmarco-passage-v2/trec-dl-2022/judged"
]


def dataset_to_filename(dataset):
    return dataset.replace("/", "-") + ".run"


def load_per_query_ndcg(run_path, dataset_name):
    """Load a run file and compute per-query nDCG@10."""
    run_df = pd.read_csv(
        run_path, sep=r"\s+", header=None,
        names=["query", "q0", "docid", "rank", "score", "system"],
        dtype={"query": str, "docid": str},
    )
    run = trectools.TrecRun()
    run.run_data = run_df

    qrels_df = pd.DataFrame(ir_datasets.load(dataset_name).qrels_iter())
    qrels_df = qrels_df.rename(
        {"query_id": "query", "doc_id": "docid", "relevance": "rel", "iteration": "q0"},
        axis=1,
    ).astype({"query": str, "docid": str})
    qrels = trectools.TrecQrel()
    qrels.qrels_data = qrels_df

    ev = trectools.TrecEval(run, qrels)
    per_query = ev.get_ndcg(depth=10, per_query=True)
    per_query = per_query.iloc[:, -1].fillna(0)
    per_query.index = per_query.index.astype(str)
    return per_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs=2, required=True, metavar=("NAME", "DIR"),
                        help="Baseline model: name and directory")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Models to compare against baseline. Alternating: name1 dir1 name2 dir2 ...")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="Equivalence margin (default: 0.02)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Significance level. Default: 0.05 / num_tests (Bonferroni)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    # Parse baseline
    baseline = (args.baseline[0], Path(args.baseline[1]))

    # Parse model name/dir pairs
    if len(args.models) % 2 != 0:
        parser.error("--models requires alternating name and directory pairs")
    models = []
    for i in range(0, len(args.models), 2):
        models.append((args.models[i], Path(args.models[i + 1])))

    all_models = [baseline] + models
    # num_tests = len(models) * len(args.datasets)
    num_tests = len(models)
    alpha = args.alpha if args.alpha else 0.05 / num_tests

    print(f"Baseline: {baseline[0]}")
    print(f"Models: {', '.join(name for name, _ in models)}")
    print(f"Datasets: {len(args.datasets)}")
    print(f"Comparisons per dataset: {len(models)} (each model vs baseline)")
    print(f"Total tests: {num_tests}")
    print(f"Equivalence margin: ±{args.epsilon}")
    print(f"Significance level: {alpha:.4f} (Bonferroni: 0.05 / {num_tests})")
    print()

    # First print mean nDCG@10 per model per dataset
    print("=" * 80)
    print("Mean nDCG@10")
    print("=" * 80)
    header = f"{'Dataset':<35}"
    for name, _ in all_models:
        header += f" {name:>12}"
    print(header)
    print("-" * 80)

    # Cache scores for reuse in TOST
    score_cache = {}
    for dataset in args.datasets:
        filename = dataset_to_filename(dataset)
        row = f"{dataset:<35}"
        for name, model_dir in all_models:
            run_path = model_dir / filename
            if not run_path.exists():
                row += f" {'MISSING':>12}"
                continue
            scores = load_per_query_ndcg(run_path, dataset)
            score_cache[(name, dataset)] = scores
            row += f" {scores.mean():>12.3f}"
        print(row)

    # TOST results
    print()
    print("=" * 110)
    print("Paired TOST Equivalence Tests (each model vs baseline)")
    print("=" * 110)
    print(f"{'Dataset':<35} {'Comparison':<30} {'#Q':>5} {'Diff':>8} "
          f"{'p(TOST)':>10} {'Equiv?':>8}")
    print("-" * 110)

    baseline_name = baseline[0]
    for dataset in args.datasets:
        for name, _ in models:
            key_model = (name, dataset)
            key_baseline = (baseline_name, dataset)

            if key_model not in score_cache or key_baseline not in score_cache:
                comparison = f"{name} vs {baseline_name}"
                print(f"{dataset:<35} {comparison:<30} {'MISSING':>5}")
                continue

            scores_model = score_cache[key_model]
            scores_baseline = score_cache[key_baseline]

            # Align on common queries
            common = scores_model.index.intersection(scores_baseline.index)
            vals_model = scores_model.loc[common].values
            vals_baseline = scores_baseline.loc[common].values

            p_tost, (t1, p1, df1), (t2, p2, df2) = ttost_paired(
                vals_model, vals_baseline, low=-args.epsilon, upp=args.epsilon
            )

            mean_diff = (vals_model - vals_baseline).mean()
            equiv = "Yes" if p_tost < alpha else "No"
            comparison = f"{name} vs {baseline_name}"

            print(f"{dataset:<35} {comparison:<30} {len(common):>5} {mean_diff:>+8.4f} "
                  f"{p_tost:>10.6f} {equiv:>8}")

    print()
    print(f"Equivalent = p(TOST) < {alpha:.4f} (Bonferroni-corrected)")
    print(f"H0: |mean(model - baseline)| >= {args.epsilon}  (not equivalent)")
    print(f"H1: |mean(model - baseline)| <  {args.epsilon}  (equivalent)")

    trec_datasets = [d for d in args.datasets if "trec-dl" in d]
    if trec_datasets:
        print()
        print("=" * 110)
        print("Micro-averaged TREC DL TOST")
        print("=" * 110)
        print(f"{'Comparison':<40} {'#Q':>5} {'Sparse':>8} {'Full':>8} {'Diff':>8} "
              f"{'p(TOST)':>10} {'Equiv?':>8}")
        print("-" * 110)

        baseline_name = baseline[0]
        for name, _ in models:
            all_model = []
            all_baseline = []
            for dataset in trec_datasets:
                key_model = (name, dataset)
                key_baseline = (baseline_name, dataset)
                if key_model not in score_cache or key_baseline not in score_cache:
                    continue
                scores_model = score_cache[key_model]
                scores_baseline = score_cache[key_baseline]
                common = scores_model.index.intersection(scores_baseline.index)
                all_model.extend(scores_model.loc[common].values)
                all_baseline.extend(scores_baseline.loc[common].values)

            all_model = np.array(all_model)
            all_baseline = np.array(all_baseline)

            p_tost, _, _ = ttost_paired(
                all_model, all_baseline, low=-args.epsilon, upp=args.epsilon
            )
            mean_diff = (all_model - all_baseline).mean()
            equiv = "Yes" if p_tost < alpha else "No"

            print(f"{name} vs {baseline_name:<25} {len(all_model):>5} "
                  f"{all_model.mean():>8.3f} {all_baseline.mean():>8.3f} "
                  f"{mean_diff:>+8.4f} {p_tost:>10.6f} {equiv:>8}")


if __name__ == "__main__":
    main()
