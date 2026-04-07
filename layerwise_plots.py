import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Literal

def avg_rank_correlation_heatmap(
        datasets: list,
        layers: list,
        layerwise_df: pd.DataFrame,
        model_a: str,
        model_b: str,
        domain_type: Literal['all', 'in-domain', 'out-domain'] = 'all'
):
    """
    # The Metric (Color): The color of the bubble represents the mean difference in
    Kendall τ between Model A's layer and Model B's layer. Darker colors mean higher alignment (lower difference).
    # The Deviation (Size): The size of the bubble represents the consistency (inverse of the standard deviation).
    """
    diff_records = []
    for dset in datasets:
        dset_cut_df = layerwise_df[layerwise_df['Dataset Name'] == dset]
        tau_a = dset_cut_df[dset_cut_df['Model Name'] == model_a].set_index('Layer')['Kendall τ']
        tau_b = dset_cut_df[dset_cut_df['Model Name'] == model_b].set_index('Layer')['Kendall τ']

        for layer_y in layers:
            for layer_x in layers:
                diff_records.append({
                    'Dataset': dset,
                    'Layer_A': layer_x,
                    'Layer_B': layer_y,
                    'Diff': tau_a[layer_x] - tau_b[layer_y]
                })

    diff_df = pd.DataFrame(diff_records)

    agg_df = diff_df.groupby(['Layer_A', 'Layer_B'])['Diff'].agg(['mean', 'std']).reset_index()

    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    agg_df['x_idx'] = agg_df['Layer_A'].map(layer_to_idx)
    agg_df['y_idx'] = agg_df['Layer_B'].map(layer_to_idx)

    # Map Standard Deviation to Bubble Size (Consistency)
    # We want LOW std -> HIGH consistency -> BIG bubble.
    min_std = agg_df['std'].min()
    max_std = agg_df['std'].max()

    # Inverse linear mapping: Big bubbles (800) for low deviation, small dots (50) for high deviation
    agg_df['bubble_size'] = 800 - 750 * ((agg_df['std'] - min_std) / (max_std - min_std))

    fig, ax = plt.subplots(figsize=(9, 7))

    max_abs_diff = agg_df['mean'].abs().max()

    scatter = ax.scatter(
        x=agg_df['x_idx'],
        y=agg_df['y_idx'],
        s=agg_df['bubble_size'],
        c=agg_df['mean'],
        cmap='vlag',  # Blue = Full Attn wins, White = Tie, Red = Sparse wins
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        edgecolors='gray',
        linewidth=0.5,
        alpha=0.9
    )

    # annotate main diagonal
    diagonal_data = agg_df[agg_df['Layer_A'] == agg_df['Layer_B']]

    for _, row in diagonal_data.iterrows():
        ax.text(
            x=row['x_idx'],
            y=row['y_idx'],
            s=f"{row['mean']:.3f}",
            color='black',
            va='center',
            ha='center',
            fontsize=7,
            fontweight='bold'
        )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', rotation_mode='anchor')

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    ax.set_xticks([x - 0.5 for x in range(1, len(layers))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(layers))], minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', length=0)  # Hide minor tick marks

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(f'Mean Δ Kendall τ ({model_a} - {model_b})')

    plt.title(
        f"Average Layer Alignment Across {domain_type} Datasets\n"
        "Color = Mean Performance Difference | Size = Consistency (Low Std. Dev)",
        pad=15
    )
    plt.xlabel(f"{model_a} Layer")
    plt.ylabel(f"{model_b} Layer")

    plt.tight_layout()
    plt.savefig(f"average_bubble_heatmap_{domain_type}_alignment.png", dpi=300)
    plt.close()

def avg_ndcg(
    models: list,
    layers: list,
    layerwise_df: pd.DataFrame,
    model_colors: dict,
    domain_type: Literal['all', 'in-domain', 'out-domain'] = 'all'
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        model_df = layerwise_df[layerwise_df['Model Name'] == model]

        means = []
        stds = []

        for layer in layers:
            layer_values = model_df[model_df['Layer'] == layer]['nDCG@10']

            means.append(layer_values.mean())
            stds.append(layer_values.std())

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            layers,
            means,
            color=model_colors[model],
            label=model,
            marker='o',
            linewidth=2
        )

        ax.fill_between(
            layers,
            means - stds,
            means + stds,
            color=model_colors[model],
            alpha=0.15,
            edgecolor='none'
        )

    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=8)
    plt.title(f"Average Layer-wise nDCG@10 Across {domain_type} Datasets", wrap=True)
    plt.ylabel("Average nDCG@10")

    plt.tight_layout()
    plt.savefig(f"average_ndcg_layerwise_{domain_type}_with_deviation.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    model_colors = {
        'Reproduced Sparse 512 w=4': '#1f77b4',  # Blue
        'Author Sparse 512 w=4': '#ff7f0e',  # Orange
        'Full Attention': '#2ca02c'  # Green
    }

    layerwise_df = pd.read_csv(os.path.join(os.getcwd(), "layerwise.csv"))

    datasets = layerwise_df['Dataset Name'].unique().tolist()
    models = layerwise_df['Model Name'].unique().tolist()

    # layers are the same for all models because they share a base model
    layers = layerwise_df['Layer'].unique().tolist()

    for dset in datasets:
        dset_cut_df = layerwise_df[layerwise_df['Dataset Name'] == dset]

        fig, ax = plt.subplots()

        for model in models:
            ax.plot(layers,
                    dset_cut_df[dset_cut_df['Model Name'] == model]['nDCG@10'],
                    color=model_colors[model], label=model)

        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=8)
        plt.title(f"Layer-wise NDCG@10 on dataset {dset.replace('/', '-')}", wrap=True)
        plt.ylabel("nDCG@10")
        plt.savefig(f"{dset.replace('/', '-')}_ndcg_layerwise.png", dpi=300)
        plt.close()

    # Average NDCG across all datasets, per model
    avg_ndcg(models, layers, layerwise_df, model_colors)

    # Average NDCG across in-out domain, per model
    indomain_datasets = [d for d in datasets if "beir" not in d]
    layerwise_df_indomain = layerwise_df[layerwise_df['Dataset Name'].isin(indomain_datasets)]
    outdomain_datasets = [d for d in datasets if "beir" in d]
    layerwise_df_outdomain = layerwise_df[layerwise_df['Dataset Name'].isin(outdomain_datasets)]

    avg_ndcg(models, layers, layerwise_df_indomain, model_colors, 'in-domain')
    avg_ndcg(models, layers, layerwise_df_outdomain, model_colors, 'out-domain')

    ################################################
    ################################################
    ################################################
    ################################################

    # Kendall's tau heatmaps

    model_a = 'Reproduced Sparse 512 w=4'
    model_b = 'Full Attention'

    for dset in datasets:
        dset_cut_df = layerwise_df[layerwise_df['Dataset Name'] == dset]

        # Extract the Kendall τ values for both models as Series indexed by Layer
        tau_a = dset_cut_df[dset_cut_df['Model Name'] == model_a].set_index('Layer')['Kendall τ']
        tau_b = dset_cut_df[dset_cut_df['Model Name'] == model_b].set_index('Layer')['Kendall τ']

        heatmap_data = pd.DataFrame(index=layers, columns=layers, dtype=float)

        for layer_y in layers:
            for layer_x in layers:
                heatmap_data.loc[layer_y, layer_x] = tau_a[layer_x] - tau_b[layer_y]

        plt.figure(figsize=(8, 6))

        # center=0 ensures that identical performance is neutral/white
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="vlag",  # Blue (negative) to Red (positive)
            center=0,
            cbar_kws={'label': f'Δ Kendall τ ({model_a} - {model_b})'},
            linewidths=0.5,
            linecolor='white'
        )

        ax.invert_yaxis()

        plt.title(f"Layer Performance Alignment (Kendall τ)\n{dset.replace('/', '-')}", pad=15, wrap=True)
        plt.xlabel(f"{model_a} Layer")
        plt.ylabel(f"{model_b} Layer")

        plt.xticks(rotation=30, ha="right", rotation_mode="anchor", fontsize=8)
        plt.yticks(rotation=0)  # Keep y-axis text horizontal

        plt.tight_layout()
        plt.savefig(f"{dset.replace('/', '-')}_heatmap_tau.png", dpi=300)
        plt.close()

    # create bubble heatmaps (Size + Color)
    avg_rank_correlation_heatmap(datasets, layers, layerwise_df, model_a, model_b)
    avg_rank_correlation_heatmap(indomain_datasets, layers, layerwise_df_indomain, model_a, model_b, 'in-domain')
    avg_rank_correlation_heatmap(outdomain_datasets, layers, layerwise_df_outdomain, model_a, model_b, 'out-domain')
