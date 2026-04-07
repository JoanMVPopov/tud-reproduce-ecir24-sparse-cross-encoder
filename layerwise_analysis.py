"""
Layer-wise prediction analysis for sparse cross-encoder models.
Compares how each intermediate layer's predictions relate to the final layer.

For each layer, applies the final classifier head to that layer's CLS output
and computes nDCG@10 and Kendall's tau correlation with the final layer's ranking.

Usage:
    python layerwise_analysis.py \
        --models \
            "Our Sparse" YoanPopov/reproduced-ecir24-sparse-512 \
            "Sparse CE" webis/sparse-cross-encoder-4-512 \
            "Full Attn" cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --datasets beir/hotpotqa/test beir/nq beir/dbpedia-entity/test \
        --run_dir data/baseline-runs/bm25 \
        --output_csv layerwise_results.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import ir_datasets
import numpy as np
import pandas as pd
import torch
import trectools
from scipy.stats import kendalltau

from sparse_cross_encoder.data.datamodule import SparseCrossEncoderDataModule
from sparse_cross_encoder.data.ir_dataset_utils import load as load_ir_dataset
from sparse_cross_encoder.model.sparse_cross_encoder import (
    SparseCrossEncoderConfig,
    SparseCrossEncoderModelForSequenceClassification,
    SparseCrossEncoderPreTrainedModel,
)

import transformers

DATASETS = [
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "msmarco-passage-v2/trec-dl-2021/judged",
    "msmarco-passage-v2/trec-dl-2022/judged",
    "beir/hotpotqa/test",
    "beir/nq",
    "beir/dbpedia-entity/test"
]


def dataset_to_filename(dataset):
    return dataset.replace("/", "-") + ".run"


def load_model(model_name_or_path):
    """Load model and tokenizer, return model with access to intermediate layers."""
    base_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    SparseCrossEncoderModelForSequenceClassification.base_model_prefix = base_config.model_type
    SparseCrossEncoderPreTrainedModel.base_model_prefix = base_config.model_type
    SparseCrossEncoderConfig.model_type = base_config.model_type

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model_config = SparseCrossEncoderConfig.from_pretrained(model_name_or_path)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.cls_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    model_config.sep_token_id = tokenizer.sep_token_id

    model = SparseCrossEncoderModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def forward_all_layers(model, query_input_ids, doc_input_ids):
    """Run forward pass and get logits from each layer's CLS output."""
    with torch.no_grad():
        output = getattr(model, model.config.model_type).forward(
            query_input_ids, doc_input_ids, output_hidden_states=True
        )

    # output.hidden_states is a tuple of HiddenStates (one per layer + input embeddings)
    # Each HiddenStates has .cls, .query, .doc
    # Apply pooler (if exists) and classifier to each layer's CLS
    all_layer_logits = []
    encoder_model = getattr(model, model.config.model_type)

    for layer_idx, hidden_state in enumerate(output.hidden_states):
        if encoder_model.pooler is not None:
            cls_output = encoder_model.pooler(hidden_state.cls)
        else:
            cls_output = hidden_state.cls
        logits = model.classifier(cls_output)[..., 0, 0]
        layer_logits = [
            logits[batch_idx, : len(doc_input_ids[batch_idx])]
            for batch_idx in range(logits.shape[0])
        ]
        all_layer_logits.append(layer_logits)

    return all_layer_logits  # list of num_layers+1, each containing batch of logit tensors


def compute_ndcg(run_df, dataset_name):
    """Compute per-query nDCG@10."""
    run = trectools.TrecRun()
    run.run_data = run_df.sort_values(
        ["query", "score", "docid"], ascending=[True, False, False]
    ).reset_index(drop=True)

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
    return per_query.mean()


def run_layerwise_analysis(model, tokenizer, dataset_name, run_dir, batch_size=2, max_length=512):
    """Run layer-wise analysis for a single model and dataset."""
    run_path = run_dir / dataset_to_filename(dataset_name)
    if not run_path.exists():
        print(f"  WARNING: missing {run_path}")
        return None

    # Set up data
    datamodule = SparseCrossEncoderDataModule(
        model_name_or_path=tokenizer.name_or_path,
        ir_dataset_path=run_path,
        truncate=True,
        max_length=max_length,
        batch_size=batch_size,
        depth=100,
    )
    datamodule.setup(stage="predict")
    dataloader = datamodule.predict_dataloader()

    num_layers = model.config.num_hidden_layers + 1  # +1 for input embeddings
    layer_results = [[] for _ in range(num_layers)]  # query_id, doc_id, score per layer
    final_layer_idx = num_layers - 1

    device = next(model.parameters()).device

    for batch_idx, batch in enumerate(dataloader):
        query_input_ids = [t.to(device) for t in batch["query_input_ids"]]
        doc_input_ids = [[t.to(device) for t in docs] for docs in batch["doc_input_ids"]]

        all_layer_logits = forward_all_layers(model, query_input_ids, doc_input_ids)

        for layer_idx in range(num_layers):
            for b_idx in range(len(batch["query_id"])):
                query_id = batch["query_id"][b_idx]
                doc_ids = batch["doc_ids"][b_idx]
                scores = all_layer_logits[layer_idx][b_idx].cpu().numpy()
                for doc_id, score in zip(doc_ids, scores):
                    layer_results[layer_idx].append({
                        "query": str(query_id),
                        "q0": "Q0",
                        "docid": str(doc_id),
                        "rank": 0,
                        "score": float(score),
                        "system": f"layer_{layer_idx}",
                    })

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}")

    results = {}
    final_run_df = pd.DataFrame(layer_results[final_layer_idx])
    # Assign ranks
    final_run_df["rank"] = final_run_df.groupby("query")["score"].rank(ascending=False, method='first').astype(int)

    for layer_idx in range(num_layers):
        run_df = pd.DataFrame(layer_results[layer_idx])
        run_df["rank"] = run_df.groupby("query")["score"].rank(ascending=False, method='first').astype(int)

        ndcg = compute_ndcg(run_df, dataset_name)

        # Kendall's tau with final layer (per query, then average)
        taus = []
        for query_id in run_df["query"].unique():
            layer_scores = run_df[run_df["query"] == query_id].set_index("docid")["score"]
            final_scores = final_run_df[final_run_df["query"] == query_id].set_index("docid")["score"]
            common = layer_scores.index.intersection(final_scores.index)
            if len(common) > 1:
                tau, _ = kendalltau(layer_scores.loc[common].values, final_scores.loc[common].values)
                if not np.isnan(tau):
                    taus.append(tau)
        avg_tau = np.mean(taus) if taus else 0.0

        layer_name = f"Layer {layer_idx}" if layer_idx > 0 else "Embeddings"
        results[layer_idx] = {
            "layer": layer_name,
            "ndcg@10": ndcg,
            "kendall_tau": avg_tau,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Alternating: name1 path1 name2 path2 ...")
    parser.add_argument("--run_dir", type=str, default="data/baseline-runs/bm25")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    if len(args.models) % 2 != 0:
        parser.error("--models requires alternating name and path pairs")
    models = []
    for i in range(0, len(args.models), 2):
        models.append((args.models[i], args.models[i + 1]))

    run_dir = Path(args.run_dir)
    all_results = []

    for model_name, model_path in models:
        print(f"\nLoading model: {model_name} ({model_path})")
        model, tokenizer = load_model(model_path)
        # num_layers = model.config.num_hidden_layers

        for dataset in DATASETS:
            print(f"  Dataset: {dataset}")
            results = run_layerwise_analysis(
                model, tokenizer, dataset, run_dir,
                batch_size=args.batch_size,
            )
            if results is None:
                continue

            print(f"  {'Layer':<15} {'nDCG@10':>10} {'Kendall τ':>12}")
            print(f"  {'-'*40}")
            for layer_idx, layer_result in results.items():
                print(f"  {layer_result['layer']:<15} {layer_result['ndcg@10']:>10.3f} "
                      f"{layer_result['kendall_tau']:>12.3f}")
                all_results.append({
                    "model": model_name,
                    "dataset": dataset,
                    **layer_result,
                })

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    if args.output_csv and all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
