"""
Evaluation script for comparing runs and generating visualizations.
Independent script that fetches metrics from WandB and generates comparison reports.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import os

import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (optional)")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project (optional)")
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run history, summary, and config from WandB.
    
    Returns:
        Dict with 'config', 'summary', 'history' keys
    """
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"
    
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Warning: Could not fetch WandB run {run_path}: {e}")
        return None
    
    config = dict(run.config)
    summary = dict(run.summary)
    history = run.history()
    
    return {
        "config": config,
        "summary": summary,
        "history": history.to_dict('records') if not history.empty else []
    }


def load_local_metrics(results_dir: Path, run_id: str) -> Dict:
    """
    Load metrics from local file as fallback.
    """
    metrics_file = results_dir / run_id / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {}


def export_per_run_metrics(results_dir: Path, run_id: str, metrics: Dict):
    """
    Export per-run metrics to results_dir/{run_id}/metrics.json
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics to {metrics_file}")


def create_per_run_figures(results_dir: Path, run_id: str, metrics: Dict, run_data: Dict):
    """
    Create per-run figures.
    
    For inference tasks:
    - Bar chart of main metrics
    - Confidence distribution (if available)
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Main metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metric_names = ["accuracy", "confidently_wrong_rate", "average_samples_used"]
    metric_values = [metrics.get(m, 0.0) for m in metric_names]
    metric_labels = ["Accuracy", "Conf. Wrong Rate", "Avg Samples"]
    
    ax.bar(metric_labels, metric_values, color=["green", "red", "blue"])
    ax.set_ylabel("Value")
    ax.set_title(f"Run: {run_id}")
    ax.set_ylim(0, max(metric_values) * 1.2)
    
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    
    fig_path = run_dir / "metrics_bar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created figure: {fig_path}")
    
    # Figure 2: Verification pass rate (if VMSCA)
    if "verification_pass_rate" in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        vpr = metrics["verification_pass_rate"]
        ax.bar(["Verification Pass Rate"], [vpr], color="purple")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"VMSCA Verification - {run_id}")
        ax.text(0, vpr + 0.02, f"{vpr:.3f}", ha="center", fontsize=12)
        
        fig_path = run_dir / "verification_rate.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Created figure: {fig_path}")


def create_comparison_figures(results_dir: Path, run_ids: List[str], all_metrics: Dict):
    """
    Create comparison figures across all runs.
    """
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accuracies = [all_metrics[rid].get("accuracy", 0.0) for rid in run_ids]
    colors = ["green" if "proposed" in rid else "blue" for rid in run_ids]
    
    ax.bar(run_ids, accuracies, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 1.0)
    plt.xticks(rotation=45, ha="right")
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    
    fig_path = comp_dir / "accuracy_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created figure: {fig_path}")
    
    # Figure 2: Confidently-wrong rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cw_rates = [all_metrics[rid].get("confidently_wrong_rate", 0.0) for rid in run_ids]
    
    ax.bar(run_ids, cw_rates, color="red", alpha=0.7)
    ax.set_ylabel("Confidently Wrong Rate")
    ax.set_title("Confidently Wrong Rate Comparison")
    ax.set_ylim(0, max(cw_rates) * 1.2 if cw_rates else 1.0)
    plt.xticks(rotation=45, ha="right")
    
    for i, v in enumerate(cw_rates):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    
    fig_path = comp_dir / "confidently_wrong_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created figure: {fig_path}")
    
    # Figure 3: Average samples used
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_samples = [all_metrics[rid].get("average_samples_used", 0.0) for rid in run_ids]
    
    ax.bar(run_ids, avg_samples, color="blue", alpha=0.7)
    ax.set_ylabel("Avg Samples Used")
    ax.set_title("Sampling Efficiency Comparison")
    plt.xticks(rotation=45, ha="right")
    
    for i, v in enumerate(avg_samples):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    
    fig_path = comp_dir / "samples_used_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Created figure: {fig_path}")


def export_aggregated_metrics(results_dir: Path, run_ids: List[str], all_metrics: Dict, primary_metric: str = "accuracy"):
    """
    Export aggregated metrics with best proposed/baseline comparison.
    """
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate proposed and baseline
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]
    baseline_runs = [rid for rid in run_ids if "comparative" in rid]
    
    proposed_scores = [all_metrics[rid].get(primary_metric, 0.0) for rid in proposed_runs]
    baseline_scores = [all_metrics[rid].get(primary_metric, 0.0) for rid in baseline_runs]
    
    best_proposed = max(proposed_scores) if proposed_scores else 0.0
    best_baseline = max(baseline_scores) if baseline_scores else 0.0
    gap = best_proposed - best_baseline
    
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": {rid: all_metrics[rid] for rid in run_ids},
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "proposed_runs": proposed_runs,
        "baseline_runs": baseline_runs
    }
    
    agg_file = comp_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported aggregated metrics to {agg_file}")
    print(f"\n=== Summary ===")
    print(f"Best Proposed ({primary_metric}): {best_proposed:.4f}")
    print(f"Best Baseline ({primary_metric}): {best_baseline:.4f}")
    print(f"Gap: {gap:.4f}")


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    # Determine WandB config
    # Try to load from first run's config file or use args
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project
    
    if wandb_entity is None or wandb_project is None:
        # Try to infer from environment or config
        wandb_entity = os.environ.get("WANDB_ENTITY", "airas")
        wandb_project = os.environ.get("WANDB_PROJECT", "2026-02-11")
    
    print(f"Using WandB: {wandb_entity}/{wandb_project}")
    
    # Fetch data for each run
    all_metrics = {}
    all_run_data = {}
    
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Try to fetch from WandB
        run_data = fetch_run_data(wandb_entity, wandb_project, run_id)
        
        if run_data is not None:
            metrics = run_data["summary"]
            all_run_data[run_id] = run_data
        else:
            # Fallback to local metrics
            print(f"Falling back to local metrics for {run_id}")
            metrics = load_local_metrics(results_dir, run_id)
            all_run_data[run_id] = {"config": {}, "summary": metrics, "history": []}
        
        all_metrics[run_id] = metrics
        
        # Export per-run metrics
        export_per_run_metrics(results_dir, run_id, metrics)
        
        # Create per-run figures
        create_per_run_figures(results_dir, run_id, metrics, all_run_data[run_id])
    
    # Create comparison figures
    print("\n=== Creating comparison figures ===")
    create_comparison_figures(results_dir, run_ids, all_metrics)
    
    # Export aggregated metrics
    export_aggregated_metrics(results_dir, run_ids, all_metrics, primary_metric="accuracy")
    
    print("\nEvaluation completed successfully")


if __name__ == "__main__":
    main()
