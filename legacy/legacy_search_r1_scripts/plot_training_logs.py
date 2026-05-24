#!/usr/bin/env python3
"""
Parse Memory-R1 training logs and generate plots.

Usage:
    python plot_training_logs.py [--log-dir log/] [--output-dir plots/]
"""
import argparse
import re
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_log_file(filepath: str) -> dict:
    """Parse a training log file and extract metrics per step."""
    metrics = defaultdict(list)
    steps = []
    val_steps = []
    val_metrics = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            line = re.sub(r'\(main_task pid=\d+\)\s*', '', line)
            line = line.strip()

            # Match validation lines: step:N - val/...
            val_match = re.match(r'step:(\d+)\s*-\s*(val/\S+):([\d.]+)', line)
            if val_match:
                step = int(val_match.group(1))
                key = val_match.group(2)
                value = float(val_match.group(3))
                val_steps.append(step)
                val_metrics[key].append((step, value))
                continue

            # Match training step lines: step:N - metric:value - metric:value ...
            step_match = re.match(r'step:(\d+)\s*-\s*(.*)', line)
            if step_match:
                step = int(step_match.group(1))
                rest = step_match.group(2)

                # Parse all key:value pairs
                pairs = re.findall(r'(\S+):([-\d.]+)', rest)
                if pairs:
                    steps.append(step)
                    for key, value in pairs:
                        try:
                            metrics[key].append((step, float(value)))
                        except ValueError:
                            pass

    return {
        "steps": steps,
        "metrics": dict(metrics),
        "val_steps": val_steps,
        "val_metrics": dict(val_metrics),
    }


def smooth(values, window=10):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_metric_group(ax, data, metric_keys, title, ylabel,
                      smooth_window=10, val_data=None, val_keys=None):
    """Plot a group of related metrics on one axis."""
    has_data = False
    for key in metric_keys:
        if key in data["metrics"]:
            pairs = data["metrics"][key]
            steps = [p[0] for p in pairs]
            values = [p[1] for p in pairs]
            if len(values) > 0:
                has_data = True
                label = key.split("/")[-1] if "/" in key else key
                ax.plot(steps, values, alpha=0.3, linewidth=0.5)
                if len(values) > smooth_window:
                    smoothed = smooth(values, smooth_window)
                    s_steps = steps[smooth_window - 1:]
                    ax.plot(s_steps, smoothed, label=f"{label} (avg)", linewidth=1.5)
                else:
                    ax.plot(steps, values, label=label, linewidth=1.5)

    # Overlay validation points
    if val_data and val_keys:
        for key in val_keys:
            if key in val_data:
                pairs = val_data[key]
                v_steps = [p[0] for p in pairs]
                v_values = [p[1] for p in pairs]
                if v_values:
                    has_data = True
                    label = key.replace("val/", "val: ")
                    ax.scatter(v_steps, v_values, s=60, zorder=5,
                               marker='*', label=label)

    if has_data:
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    return has_data


def generate_plots(data, stage_name, output_dir):
    """Generate all plots for a training run."""
    os.makedirs(output_dir, exist_ok=True)

    # Define metric groups
    plot_configs = [
        {
            "filename": "reward_score",
            "title": f"{stage_name}: Reward & Score",
            "ylabel": "Score",
            "keys": ["critic/score/mean", "critic/score/max", "critic/score/min"],
            "val_keys": [k for k in data["val_metrics"] if "score" in k],
            "smooth": 15,
        },
        {
            "filename": "rewards_detail",
            "title": f"{stage_name}: Rewards",
            "ylabel": "Reward",
            "keys": ["critic/rewards/mean", "critic/rewards/max", "critic/rewards/min"],
            "smooth": 15,
        },
        {
            "filename": "advantages",
            "title": f"{stage_name}: Advantages",
            "ylabel": "Advantage",
            "keys": ["critic/advantages/mean", "critic/advantages/max", "critic/advantages/min"],
            "smooth": 15,
        },
        {
            "filename": "policy_loss",
            "title": f"{stage_name}: Policy Gradient Loss",
            "ylabel": "Loss",
            "keys": ["actor/pg_loss"],
            "smooth": 15,
        },
        {
            "filename": "kl_divergence",
            "title": f"{stage_name}: KL Divergence",
            "ylabel": "KL",
            "keys": ["actor/kl_loss", "actor/ppo_kl"],
            "smooth": 15,
        },
        {
            "filename": "entropy",
            "title": f"{stage_name}: Entropy",
            "ylabel": "Entropy",
            "keys": ["actor/entropy_loss"],
            "smooth": 15,
        },
        {
            "filename": "grad_norm",
            "title": f"{stage_name}: Gradient Norm",
            "ylabel": "Grad Norm",
            "keys": ["actor/grad_norm"],
            "smooth": 15,
        },
        {
            "filename": "clip_fraction",
            "title": f"{stage_name}: PPO Clip Fraction",
            "ylabel": "Clip Fraction",
            "keys": ["actor/pg_clipfrac"],
            "smooth": 15,
        },
        {
            "filename": "response_length",
            "title": f"{stage_name}: Response Length",
            "ylabel": "Tokens",
            "keys": ["response_length/mean", "response_length/max", "response_length/min"],
            "smooth": 15,
        },
        {
            "filename": "prompt_length",
            "title": f"{stage_name}: Prompt Length",
            "ylabel": "Tokens",
            "keys": ["prompt_length/mean", "prompt_length/max", "prompt_length/min"],
            "smooth": 15,
        },
        {
            "filename": "timing",
            "title": f"{stage_name}: Step Timing",
            "ylabel": "Seconds",
            "keys": ["timing_s/step", "timing_s/ref", "timing_s/update_actor"],
            "smooth": 15,
        },
        {
            "filename": "learning_rate",
            "title": f"{stage_name}: Learning Rate",
            "ylabel": "LR",
            "keys": ["actor/lr"],
            "smooth": 1,
        },
    ]

    # Individual plots
    for cfg in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 5))
        val_keys = cfg.get("val_keys", None)
        has_data = plot_metric_group(
            ax, data, cfg["keys"], cfg["title"], cfg["ylabel"],
            smooth_window=cfg.get("smooth", 10),
            val_data=data["val_metrics"],
            val_keys=val_keys,
        )
        if has_data:
            plt.tight_layout()
            path = os.path.join(output_dir, f"{cfg['filename']}.png")
            fig.savefig(path, dpi=150)
            print(f"  Saved: {path}")
        plt.close(fig)

    # Summary dashboard (2x3 grid of most important metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{stage_name} — Training Dashboard", fontsize=14, fontweight='bold')

    dashboard_configs = [
        (axes[0, 0], ["critic/score/mean", "critic/score/max"], "Reward Score", "Score",
         [k for k in data["val_metrics"] if "score" in k]),
        (axes[0, 1], ["actor/pg_loss"], "Policy Loss", "Loss", None),
        (axes[0, 2], ["actor/kl_loss", "actor/ppo_kl"], "KL Divergence", "KL", None),
        (axes[1, 0], ["actor/grad_norm"], "Gradient Norm", "Norm", None),
        (axes[1, 1], ["response_length/mean"], "Response Length", "Tokens", None),
        (axes[1, 2], ["actor/pg_clipfrac"], "Clip Fraction", "Fraction", None),
    ]

    for ax, keys, title, ylabel, vk in dashboard_configs:
        plot_metric_group(ax, data, keys, title, ylabel,
                          smooth_window=15, val_data=data["val_metrics"], val_keys=vk)

    plt.tight_layout()
    path = os.path.join(output_dir, "dashboard.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def print_summary(data, stage_name):
    """Print key statistics."""
    print(f"\n{'='*60}")
    print(f"  {stage_name} — Summary Statistics")
    print(f"{'='*60}")

    n_steps = len(data["steps"])
    print(f"  Total training steps: {n_steps}")

    key_metrics = [
        ("critic/score/mean", "Reward (mean)"),
        ("critic/score/max", "Reward (max)"),
        ("actor/pg_loss", "PG Loss"),
        ("actor/grad_norm", "Grad Norm"),
        ("actor/ppo_kl", "PPO KL"),
        ("actor/pg_clipfrac", "Clip Fraction"),
        ("response_length/mean", "Response Length (mean)"),
    ]

    for key, label in key_metrics:
        if key in data["metrics"]:
            values = [p[1] for p in data["metrics"][key]]
            if values:
                first_10 = values[:min(10, len(values))]
                last_10 = values[-min(10, len(values)):]
                print(f"\n  {label}:")
                print(f"    First 10 avg: {np.mean(first_10):.4f}")
                print(f"    Last 10 avg:  {np.mean(last_10):.4f}")
                print(f"    Overall:      min={min(values):.4f}  max={max(values):.4f}  mean={np.mean(values):.4f}")

    # Validation scores
    if data["val_metrics"]:
        print(f"\n  Validation scores:")
        for key, pairs in data["val_metrics"].items():
            for step, val in pairs:
                print(f"    Step {step:>4d}: {key} = {val:.4f}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot Memory-R1 training logs")
    parser.add_argument("--log-dir", default="log/", help="Directory containing .log files")
    parser.add_argument("--output-dir", default="plots/", help="Directory for output plots")
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir

    # Find training logs (answer agent and manager)
    log_files = {}
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        if "answer-agent" in fname:
            log_files[fname] = "Answer Agent (Stage 1)"
        elif "manager" in fname:
            log_files[fname] = "Memory Manager (Stage 2)"

    if not log_files:
        print(f"No training log files found in {log_dir}")
        print("Looking for files matching *answer-agent*.log or *manager*.log")
        return

    for fname, stage_name in log_files.items():
        filepath = os.path.join(log_dir, fname)
        print(f"\nProcessing: {filepath}")

        data = parse_log_file(filepath)

        if not data["steps"]:
            print(f"  No training steps found in {fname}, skipping.")
            continue

        # Create stage-specific output subdirectory
        stage_dir = os.path.join(output_dir, fname.replace(".log", ""))
        print_summary(data, stage_name)
        generate_plots(data, stage_name, stage_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
